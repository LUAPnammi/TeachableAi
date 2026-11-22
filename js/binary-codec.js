// Compact weight encoding helpers (used by engines, storage and export/import).
// Float32 data is stored as raw bytes: directly in IndexedDB (typed arrays survive structured clone),
// or as base64 text inside .json exports. Both are far smaller than a JSON array of numbers.
// Assumes a little-endian CPU, which is every phone and laptop browsers run on.

const CHUNK = 0x8000;

export function f32ToBase64(arr) {
  if (!arr || !arr.length) return '';
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

export function base64ToF32(text) {
  if (!text) return new Float32Array(0);
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, Math.floor(bytes.length / 4));
}

// Accepts any of: Float32Array, ArrayBuffer, typed array, plain number array, base64 string, nullish.
// Returns a Float32Array. A Float32Array is returned as-is (not copied).
export function toFloat32(value) {
  if (value === null || value === undefined) return new Float32Array(0);
  if (value instanceof Float32Array) return value;
  if (typeof value === 'string') return base64ToF32(value);
  if (value instanceof ArrayBuffer) return new Float32Array(value);
  return Float32Array.from(value);
}
