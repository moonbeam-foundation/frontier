// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

import "forge-std/Script.sol";
import "../src/LogEmitter.sol";

contract Deploy is Script {
    function run() external {
        uint8 transferWeight = uint8(vm.envOr("TRANSFER_WEIGHT", uint256(70)));
        uint8 swapWeight = uint8(vm.envOr("SWAP_WEIGHT", uint256(20)));
        uint8 blobWeight = uint8(vm.envOr("BLOB_WEIGHT", uint256(10)));
        uint16 blobBytes = uint16(vm.envOr("BLOB_BYTES", uint256(256)));

        vm.startBroadcast();
        LogEmitter le = new LogEmitter(transferWeight, swapWeight, blobWeight, blobBytes);
        vm.stopBroadcast();

        console.log("LogEmitter deployed at:", address(le));
        console.log(
            "  transferWeight=%s swapWeight=%s blobWeight=%s", transferWeight, swapWeight, blobWeight
        );
        console.log("  blobBytes=%s", blobBytes);
    }
}
