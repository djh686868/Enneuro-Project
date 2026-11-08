"""Length-framed JSON metadata with raw NumPy array payloads."""

import json
import struct
import numpy as np

PROTOCOL_VERSION = 1
MAX_HEADER = 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024


def _read_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(min(size, 1024 * 1024))
        if not chunk:
            raise ConnectionError('connection closed during message')
        chunks.append(chunk)
        size -= len(chunk)
    return b''.join(chunks)


def send_message(sock, message, arrays=None):
    arrays = arrays or {}
    body = bytearray()
    descriptors = []
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.hasobject:
            raise ValueError('object arrays are not supported')
        descriptors.append({'name': name, 'shape': array.shape, 'dtype': array.dtype.str,
                            'offset': len(body), 'nbytes': array.nbytes})
        body.extend(array.tobytes())
    header = dict(message, protocol_version=PROTOCOL_VERSION, arrays=descriptors)
    encoded = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_HEADER or len(body) > MAX_PAYLOAD:
        raise ValueError('message exceeds size limit')
    sock.sendall(struct.pack('!IQ', len(encoded), len(body)) + encoded + body)
    return 12 + len(encoded) + len(body)


def recv_message(sock):
    header_size, body_size = struct.unpack('!IQ', _read_exact(sock, 12))
    if header_size > MAX_HEADER or body_size > MAX_PAYLOAD:
        raise ValueError('message exceeds size limit')
    message = json.loads(_read_exact(sock, header_size))
    if message.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('unsupported protocol version')
    body = _read_exact(sock, body_size)
    arrays = {}
    for item in message.pop('arrays', []):
        name, offset, nbytes = item['name'], item['offset'], item['nbytes']
        dtype = np.dtype(item['dtype'])
        shape = tuple(item['shape'])
        if (name in arrays or dtype.hasobject or offset < 0 or nbytes < 0 or
                offset + nbytes > body_size or int(np.prod(shape)) * dtype.itemsize != nbytes):
            raise ValueError('invalid array descriptor')
        arrays[name] = np.frombuffer(body, dtype=dtype, count=int(np.prod(shape)), offset=offset).reshape(shape).copy()
    return message, arrays, 12 + header_size + body_size
