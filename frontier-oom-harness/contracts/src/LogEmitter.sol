// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

/// @title LogEmitter
/// @notice Emits a configurable burst of events per call, with realistic
///         topic/data sizes, for load-testing log infrastructure.
contract LogEmitter {
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId, uint256 value);
    event Swap(
        address indexed sender,
        address indexed recipient,
        int256 amount0,
        int256 amount1,
        uint160 sqrtPriceX96,
        uint128 liquidity,
        int24 tick
    );
    event Blob(bytes32 indexed key, bytes payload);

    uint8 public immutable transferWeight;
    uint8 public immutable swapWeight;
    uint8 public immutable blobWeight;
    uint16 public immutable blobBytes;

    uint256 private _nonce;

    constructor(uint8 _transferWeight, uint8 _swapWeight, uint8 _blobWeight, uint16 _blobBytes) {
        require(_transferWeight + _swapWeight + _blobWeight > 0, "zero weights");
        require(_blobBytes <= 4096, "blob too large");
        transferWeight = _transferWeight;
        swapWeight = _swapWeight;
        blobWeight = _blobWeight;
        blobBytes = _blobBytes;
    }

    function emitBurst(uint256 count) public {
        uint256 total = uint256(transferWeight) + swapWeight + blobWeight;
        uint256 n = _nonce;

        for (uint256 i = 0; i < count; i++) {
            uint256 slot = (n + i) % total;
            if (slot < transferWeight) {
                _emitTransfer(n + i);
            } else if (slot < transferWeight + swapWeight) {
                _emitSwap(n + i);
            } else {
                _emitBlob(n + i);
            }
        }
        _nonce = n + count;
    }

    function emitLogs(uint256 n) external {
        emitBurst(n);
    }

    function _emitTransfer(uint256 seed) internal {
        address from = address(uint160(uint256(keccak256(abi.encode(seed, "from")))));
        address to = address(uint160(uint256(keccak256(abi.encode(seed, "to")))));
        uint256 tokenId = uint256(keccak256(abi.encode(seed, "tok")));
        uint256 value = uint256(keccak256(abi.encode(seed, "val")));
        emit Transfer(from, to, tokenId, value);
    }

    function _emitSwap(uint256 seed) internal {
        address sender = address(uint160(uint256(keccak256(abi.encode(seed, "s")))));
        address recipient = address(uint160(uint256(keccak256(abi.encode(seed, "r")))));
        int256 a0 = int256(uint256(keccak256(abi.encode(seed, "a0"))));
        int256 a1 = int256(uint256(keccak256(abi.encode(seed, "a1"))));
        uint160 sp = uint160(uint256(keccak256(abi.encode(seed, "sp"))));
        uint128 lq = uint128(uint256(keccak256(abi.encode(seed, "lq"))));
        int24 tk = int24(int256(uint256(keccak256(abi.encode(seed, "tk")))));
        emit Swap(sender, recipient, a0, a1, sp, lq, tk);
    }

    function _emitBlob(uint256 seed) internal {
        bytes32 key = keccak256(abi.encode(seed, "key"));
        uint256 len = blobBytes;
        bytes memory payload = new bytes(len);
        for (uint256 off = 0; off < len; off += 32) {
            bytes32 chunk = keccak256(abi.encode(seed, off));
            uint256 remaining = len - off;
            uint256 copyLen = remaining < 32 ? remaining : 32;
            for (uint256 j = 0; j < copyLen; j++) {
                payload[off + j] = chunk[j];
            }
        }
        emit Blob(key, payload);
    }
}
