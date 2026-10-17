// SPDX-License-Identifier: Apache-2.0

//! TPM 2.0 behind the TCG PC Client TIS (FIFO) interface: five 4 KiB localities of MMIO, which Linux's tpm_tis binds
//! to on a device-tree guest ("tcg,tpm-tis-mmio") and polls, as no interrupt is offered. Commands run in a host
//! backend on a thread of their own: a slow one (an RSA key generation) must not stall the vCPU that started it, so
//! the status shows no data until it is done.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::BusDevice;

pub const TPM_TIS_SIZE: u64 = 0x5000;
/// The largest command or response: TPM 2.0's MAX_COMMAND_SIZE and MAX_RESPONSE_SIZE.
pub const TPM_BUFFER_SIZE: usize = 4096;

const ACCESS: u64 = 0x00;
const INT_ENABLE: u64 = 0x08;
const INT_VECTOR: u64 = 0x0c;
const INT_STATUS: u64 = 0x10;
const INTF_CAPABILITY: u64 = 0x14;
const STS: u64 = 0x18;
const DATA_FIFO: u64 = 0x24;
const INTERFACE_ID: u64 = 0x30;
const XDATA_FIFO: u64 = 0x80;
const DID_VID: u64 = 0xf00;
const RID: u64 = 0xf04;

const ACCESS_VALID: u8 = 0x80;
const ACCESS_ACTIVE_LOCALITY: u8 = 0x20;
const ACCESS_SEIZE: u8 = 0x08;
const ACCESS_PENDING_REQUEST: u8 = 0x04;
const ACCESS_REQUEST_USE: u8 = 0x02;
// Set: no dynamic launch (TPM_HASH_START) since reset, which never happens here.
const ACCESS_ESTABLISHMENT: u8 = 0x01;

const STS_VALID: u32 = 0x80;
const STS_COMMAND_READY: u32 = 0x40;
const STS_GO: u32 = 0x20;
const STS_DATA_AVAIL: u32 = 0x10;
const STS_EXPECT: u32 = 0x08;
const STS_SELFTEST_DONE: u32 = 0x04;
const STS_RESPONSE_RETRY: u32 = 0x02;
const STS_FAMILY_TPM2: u32 = 1 << 26;

// Interface 1.3 for TPM 2.0, 64-byte transfers, a dynamic burst count, no interrupts.
const INTF_CAPS: u32 = (3 << 28) | (3 << 9);
// TIS 1.3; the rest reserved, as ones.
const INTERFACE_TIS: u32 = 0xffff_ffff;
// The reference TPM's maker, as QEMU's: the backend's engine is IBM's (libtpms).
const DID_VID_IBM: u32 = (0x0001 << 16) | 0x1014;

/// Runs the guest's TPM commands. One at a time: the device holds it behind a lock.
pub trait TpmBackend: Send {
    /// The response to `command` from the guest at `locality`.
    fn execute(&mut self, locality: u8, command: &[u8]) -> Vec<u8>;
}

/// TPM_RC_FAILURE, for a command the backend could not answer.
pub fn tpm_failure() -> Vec<u8> {
    vec![0x80, 0x01, 0, 0, 0, 10, 0, 0, 0x01, 0x01]
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Idle,
    Ready,
    Reception,
    Execution,
    Completion,
}

pub struct TpmTis {
    backend: Arc<Mutex<Box<dyn TpmBackend>>>,
    active: Option<u8>,
    requested: u8,
    state: State,
    /// The command as it arrives, then its response.
    buffer: Vec<u8>,
    read: usize,
    running: Option<JoinHandle<Vec<u8>>>,
    aborted: Option<JoinHandle<Vec<u8>>>,
    int_enable: u32,
    int_vector: u8,
}

impl TpmTis {
    pub fn new(backend: Box<dyn TpmBackend>) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            active: None,
            requested: 0,
            state: State::Idle,
            buffer: Vec::with_capacity(TPM_BUFFER_SIZE),
            read: 0,
            running: None,
            aborted: None,
            int_enable: 0,
            int_vector: 0,
        }
    }

    fn finish(&mut self) {
        if self.aborted.as_ref().is_some_and(|h| h.is_finished()) {
            self.aborted = None;
            if self.state == State::Idle {
                self.state = State::Ready;
            }
        }
        if self.state == State::Execution && self.running.as_ref().is_some_and(|h| h.is_finished()) {
            self.buffer = self.running.take().unwrap().join().unwrap_or_else(|_| tpm_failure());
            self.read = 0;
            self.state = State::Completion;
        }
    }

    /// The command's size from its header, once that has arrived.
    fn expected(&self) -> usize {
        match self.buffer.get(2..6) {
            Some(size) => u32::from_be_bytes(size.try_into().unwrap()) as usize,
            None => usize::MAX,
        }
    }

    fn sts(&mut self) -> u32 {
        self.finish();
        let (bits, burst) = match self.state {
            State::Idle | State::Execution => (STS_VALID, 0),
            State::Ready => (STS_VALID | STS_COMMAND_READY, TPM_BUFFER_SIZE),
            State::Reception => {
                let expect = if self.buffer.len() < self.expected() { STS_EXPECT } else { 0 };
                (STS_VALID | expect, TPM_BUFFER_SIZE - self.buffer.len())
            }
            State::Completion => {
                let left = self.buffer.len() - self.read;
                (STS_VALID | if left > 0 { STS_DATA_AVAIL } else { 0 }, left)
            }
        };
        bits | STS_SELFTEST_DONE | STS_FAMILY_TPM2 | ((burst.min(0xffff) as u32) << 8)
    }

    fn access(&self, locality: u8) -> u8 {
        let mut access = ACCESS_VALID | ACCESS_ESTABLISHMENT;
        if self.active == Some(locality) {
            access |= ACCESS_ACTIVE_LOCALITY;
            if self.requested != 0 {
                access |= ACCESS_PENDING_REQUEST;
            }
        } else if self.requested & (1 << locality) != 0 {
            access |= ACCESS_REQUEST_USE;
        }
        access
    }

    fn write_access(&mut self, locality: u8, value: u8) {
        if value & ACCESS_ACTIVE_LOCALITY != 0 && self.active == Some(locality) {
            self.active = (0..5u8).rev().find(|l| self.requested & (1 << l) != 0);
            if let Some(next) = self.active {
                self.requested &= !(1 << next);
            }
        }
        if value & (ACCESS_REQUEST_USE | ACCESS_SEIZE) != 0 && self.active != Some(locality) {
            if self.active.is_none() || (value & ACCESS_SEIZE != 0 && Some(locality) > self.active) {
                self.active = Some(locality);
                self.requested &= !(1 << locality);
            } else {
                self.requested |= 1 << locality;
            }
        }
    }

    fn write_sts(&mut self, value: u32) {
        self.finish();
        if value & STS_COMMAND_READY != 0 {
            // An aborted command finishes on its thread, unheard, and the device is ready again only once it has: one
            // command thread at a time, however often the guest aborts.
            if let Some(running) = self.running.take().filter(|h| !h.is_finished()) {
                self.aborted = Some(running);
            }
            self.buffer.clear();
            self.read = 0;
            self.state = if self.aborted.is_some() { State::Idle } else { State::Ready };
        } else if value & STS_GO != 0 {
            if self.state == State::Reception && self.buffer.len() >= self.expected() {
                let command = std::mem::take(&mut self.buffer);
                let backend = self.backend.clone();
                let locality = self.active.unwrap_or(0);
                self.running = Some(thread::spawn(move || {
                    backend.lock().unwrap().execute(locality, &command)
                }));
                self.state = State::Execution;
            }
        } else if value & STS_RESPONSE_RETRY != 0 && self.state == State::Completion {
            self.read = 0;
        }
    }

    fn write_fifo(&mut self, byte: u8) {
        if self.state == State::Ready {
            self.state = State::Reception;
        }
        if self.state == State::Reception
            && self.buffer.len() < TPM_BUFFER_SIZE
            && self.buffer.len() < self.expected()
        {
            self.buffer.push(byte);
        }
    }

    fn read_fifo(&mut self) -> u8 {
        self.finish();
        match self.buffer.get(self.read) {
            Some(&byte) if self.state == State::Completion => {
                self.read += 1;
                byte
            }
            _ => 0xff,
        }
    }
}

impl BusDevice for TpmTis {
    fn read(&mut self, _vcpuid: u64, offset: u64, data: &mut [u8]) {
        let locality = (offset >> 12) as u8;
        let reg = offset & 0xfff;
        let mine = self.active == Some(locality);
        if (DATA_FIFO..DATA_FIFO + 4).contains(&reg) || (XDATA_FIFO..XDATA_FIFO + 4).contains(&reg) {
            for byte in data.iter_mut() {
                *byte = if mine { self.read_fifo() } else { 0xff };
            }
            return;
        }
        let value = match reg & !3 {
            ACCESS => self.access(locality) as u32,
            INT_ENABLE => self.int_enable,
            INT_VECTOR => self.int_vector as u32,
            INT_STATUS => 0,
            INTF_CAPABILITY => INTF_CAPS,
            STS if mine => self.sts(),
            INTERFACE_ID => INTERFACE_TIS,
            DID_VID => DID_VID_IBM,
            RID => 1,
            _ => u32::MAX,
        };
        let bytes = value.to_le_bytes();
        let at = (reg & 3) as usize;
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = bytes.get(at + i).copied().unwrap_or(0xff);
        }
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, data: &[u8]) {
        let locality = (offset >> 12) as u8;
        let reg = offset & 0xfff;
        let mine = self.active == Some(locality);
        let mut word = [0u8; 4];
        let at = (reg & 3) as usize;
        for (i, &byte) in data.iter().enumerate().take(4 - at) {
            word[at + i] = byte;
        }
        let value = u32::from_le_bytes(word);
        match reg & !3 {
            ACCESS if at == 0 => self.write_access(locality, data[0]),
            INT_ENABLE if mine => self.int_enable = value,
            INT_VECTOR if mine => self.int_vector = data[0],
            STS if mine => self.write_sts(value),
            DATA_FIFO | XDATA_FIFO if mine => data.iter().for_each(|&byte| self.write_fifo(byte)),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    /// Answers a command with a header and the command's own code.
    struct Echo;
    impl TpmBackend for Echo {
        fn execute(&mut self, locality: u8, command: &[u8]) -> Vec<u8> {
            let mut response = vec![0x80, 0x01, 0, 0, 0, 15, 0, 0, 0, 0, locality];
            response.extend_from_slice(&command[6..10]);
            response
        }
    }

    fn read8(tis: &mut TpmTis, offset: u64) -> u8 {
        let mut b = [0u8];
        tis.read(0, offset, &mut b);
        b[0]
    }

    fn read32(tis: &mut TpmTis, offset: u64) -> u32 {
        let mut b = [0u8; 4];
        tis.read(0, offset, &mut b);
        u32::from_le_bytes(b)
    }

    fn write8(tis: &mut TpmTis, offset: u64, value: u8) {
        tis.write(0, offset, &[value]);
    }

    /// TPM2_GetRandom(8): 12 bytes.
    const COMMAND: [u8; 12] = [0x80, 0x01, 0, 0, 0, 12, 0, 0, 0x01, 0x7b, 0, 8];

    fn send(tis: &mut TpmTis, command: &[u8]) {
        write8(tis, STS, STS_COMMAND_READY as u8);
        assert_eq!(read32(tis, STS) & 0xff, STS_VALID | STS_COMMAND_READY | STS_SELFTEST_DONE);
        for (i, &byte) in command.iter().enumerate() {
            write8(tis, DATA_FIFO, byte);
            let expect = read32(tis, STS) & STS_EXPECT != 0;
            assert_eq!(expect, i + 1 < command.len(), "Expect after byte {i}");
        }
        write8(tis, STS, STS_GO as u8);
    }

    fn receive(tis: &mut TpmTis) -> Vec<u8> {
        let start = Instant::now();
        while read32(tis, STS) & STS_DATA_AVAIL == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "no response");
            thread::sleep(Duration::from_millis(1));
        }
        let burst = (read32(tis, STS) >> 8) & 0xffff;
        let response: Vec<u8> = (0..burst).map(|_| read8(tis, DATA_FIFO)).collect();
        assert_eq!(read32(tis, STS) & (STS_VALID | STS_DATA_AVAIL), STS_VALID, "data left");
        response
    }

    #[test]
    fn locality_is_granted_and_relinquished() {
        let mut tis = TpmTis::new(Box::new(Echo));
        assert_eq!(read8(&mut tis, ACCESS), ACCESS_VALID | ACCESS_ESTABLISHMENT);
        assert_eq!(read32(&mut tis, STS), u32::MAX, "STS of a locality not held");
        write8(&mut tis, ACCESS, ACCESS_REQUEST_USE);
        assert_eq!(
            read8(&mut tis, ACCESS) & (ACCESS_ACTIVE_LOCALITY | ACCESS_VALID | ACCESS_REQUEST_USE),
            ACCESS_ACTIVE_LOCALITY | ACCESS_VALID
        );
        write8(&mut tis, 0x3000 + ACCESS, ACCESS_REQUEST_USE);
        assert_ne!(read8(&mut tis, ACCESS) & ACCESS_PENDING_REQUEST, 0);
        assert_ne!(read8(&mut tis, 0x3000 + ACCESS) & ACCESS_REQUEST_USE, 0);
        write8(&mut tis, ACCESS, ACCESS_ACTIVE_LOCALITY);
        assert_eq!(read8(&mut tis, ACCESS) & ACCESS_ACTIVE_LOCALITY, 0);
        assert_ne!(read8(&mut tis, 0x3000 + ACCESS) & ACCESS_ACTIVE_LOCALITY, 0, "handed to the requester");
    }

    #[test]
    fn a_command_goes_to_the_backend_and_back() {
        let mut tis = TpmTis::new(Box::new(Echo));
        assert_eq!(read32(&mut tis, DID_VID), DID_VID_IBM);
        write8(&mut tis, ACCESS, ACCESS_REQUEST_USE);
        send(&mut tis, &COMMAND);
        assert_eq!(receive(&mut tis), [0x80, 0x01, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0x01, 0x7b]);
        write8(&mut tis, STS, STS_RESPONSE_RETRY as u8);
        assert_eq!(receive(&mut tis).len(), 15, "a retry reads it again");
        send(&mut tis, &COMMAND);
        assert_eq!(receive(&mut tis).len(), 15, "and the next command after it");
    }

    #[test]
    fn go_waits_for_the_whole_command() {
        let mut tis = TpmTis::new(Box::new(Echo));
        write8(&mut tis, ACCESS, ACCESS_REQUEST_USE);
        write8(&mut tis, STS, STS_COMMAND_READY as u8);
        for &byte in &COMMAND[..8] {
            write8(&mut tis, DATA_FIFO, byte);
        }
        write8(&mut tis, STS, STS_GO as u8);
        thread::sleep(Duration::from_millis(20));
        assert_eq!(read32(&mut tis, STS) & STS_DATA_AVAIL, 0);
        assert_ne!(read32(&mut tis, STS) & STS_EXPECT, 0);
    }

    #[test]
    fn an_aborted_command_holds_the_device_until_it_ends() {
        struct Gate(Arc<Mutex<bool>>);
        impl TpmBackend for Gate {
            fn execute(&mut self, locality: u8, command: &[u8]) -> Vec<u8> {
                while !*self.0.lock().unwrap() {
                    thread::sleep(Duration::from_millis(1));
                }
                Echo.execute(locality, command)
            }
        }
        let open = Arc::new(Mutex::new(false));
        let mut tis = TpmTis::new(Box::new(Gate(open.clone())));
        write8(&mut tis, ACCESS, ACCESS_REQUEST_USE);
        send(&mut tis, &COMMAND);
        for _ in 0..3 {
            write8(&mut tis, STS, STS_COMMAND_READY as u8);
            assert_eq!(read32(&mut tis, STS) & STS_COMMAND_READY, 0, "ready while the aborted command runs");
            write8(&mut tis, DATA_FIFO, COMMAND[0]);
            write8(&mut tis, STS, STS_GO as u8);
        }
        *open.lock().unwrap() = true;
        let start = Instant::now();
        while read32(&mut tis, STS) & STS_COMMAND_READY == 0 {
            assert!(start.elapsed() < Duration::from_secs(5), "never ready again");
            thread::sleep(Duration::from_millis(1));
        }
        send(&mut tis, &COMMAND);
        assert_eq!(receive(&mut tis).len(), 15, "the next command after it");
    }

    #[test]
    fn a_failed_backend_answers_tpm_rc_failure() {
        struct Panics;
        impl TpmBackend for Panics {
            fn execute(&mut self, _: u8, _: &[u8]) -> Vec<u8> {
                panic!("backend gone")
            }
        }
        let mut tis = TpmTis::new(Box::new(Panics));
        write8(&mut tis, ACCESS, ACCESS_REQUEST_USE);
        send(&mut tis, &COMMAND);
        assert_eq!(receive(&mut tis), tpm_failure());
    }
}
