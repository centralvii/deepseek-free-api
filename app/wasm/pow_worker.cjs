const fs = require('fs');
const path = require('path');

class DeepSeekHashWasm {
  constructor() {
    this.offset = 0;
    this.cachedUint8Memory = null;
    this.cachedTextEncoder = new TextEncoder();
  }

  getCachedUint8Memory() {
    if (!this.cachedUint8Memory || !this.cachedUint8Memory.byteLength) {
      this.cachedUint8Memory = new Uint8Array(this.wasmInstance.memory.buffer);
    }
    return this.cachedUint8Memory;
  }

  encodeString(text, allocate, reallocate) {
    const strLength = text.length;
    let ptr = allocate(strLength, 1) >>> 0;
    const memory = this.getCachedUint8Memory();
    let asciiLength = 0;

    for (; asciiLength < strLength; asciiLength++) {
      if (text.charCodeAt(asciiLength) > 127) break;
      memory[ptr + asciiLength] = text.charCodeAt(asciiLength);
    }

    if (asciiLength !== strLength) {
      if (asciiLength > 0) text = text.slice(asciiLength);
      ptr = reallocate(ptr, strLength, asciiLength + text.length * 3, 1) >>> 0;
      const result = this.cachedTextEncoder.encodeInto(
        text,
        this.getCachedUint8Memory().subarray(ptr + asciiLength, ptr + asciiLength + text.length * 3)
      );
      asciiLength += result.written;
      ptr = reallocate(ptr, asciiLength + text.length * 3, asciiLength, 1) >>> 0;
    }

    this.offset = asciiLength;
    return ptr;
  }

  calculateHash(challenge, prefix, difficulty) {
    try {
      const retptr = this.wasmInstance.__wbindgen_add_to_stack_pointer(-16);

      const ptr0 = this.encodeString(
        challenge,
        this.wasmInstance.__wbindgen_export_0,
        this.wasmInstance.__wbindgen_export_1
      );
      const len0 = this.offset;

      const ptr1 = this.encodeString(
        prefix,
        this.wasmInstance.__wbindgen_export_0,
        this.wasmInstance.__wbindgen_export_1
      );
      const len1 = this.offset;

      this.wasmInstance.wasm_solve(retptr, ptr0, len0, ptr1, len1, difficulty);

      const dv = new DataView(this.wasmInstance.memory.buffer);
      const status = dv.getInt32(retptr + 0, true);
      const value = dv.getFloat64(retptr + 8, true);

      return status === 0 ? -1 : value;
    } finally {
      this.wasmInstance.__wbindgen_add_to_stack_pointer(16);
    }
  }

  async init(wasmPath) {
    const wasmBuffer = fs.readFileSync(wasmPath);
    const { instance } = await WebAssembly.instantiate(wasmBuffer, { wbg: {} });
    this.wasmInstance = instance.exports;
  }
}

async function run() {
  const args = process.argv.slice(2);
  if (args.length < 3) {
    console.error('Usage: node pow_worker.cjs <challenge> <prefix> <difficulty>');
    process.exit(1);
  }

  const [challenge, prefix, diffStr] = args;
  const difficulty = parseInt(diffStr, 10);
  const wasmPath = path.join(__dirname, 'sha3_wasm_bg.wasm');

  const solver = new DeepSeekHashWasm();
  await solver.init(wasmPath);

  const answer = solver.calculateHash(challenge, prefix, difficulty);
  console.log(answer);
}

run().catch(err => {
  console.error(err);
  process.exit(1);
});
