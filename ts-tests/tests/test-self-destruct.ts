import { expect } from "chai";

import { GENESIS_ACCOUNT, GENESIS_ACCOUNT_PRIVATE_KEY } from "./config";
import { createAndFinalizeBlock, customRequest, describeWithFrontier } from "./util";

describeWithFrontier("Frontier RPC (Constructor Revert)", (context) => {
	// ```
	// pragma solidity >=0.4.22 <0.7.0;
	//
	// contract A {
	//   constructor() {
	//     assembly { selfdestruct(0) }
	//   }
	// }
	// ```
	const CONTRACT_BYTECODE = "6080604052348015600e575f80fd5b505ffffe";

	it("should provide a tx receipt after successful deployment", async function () {
		this.timeout(15000);

		const deployContract = async () => {
			const tx = await context.web3.eth.accounts.signTransaction(
				{
					from: GENESIS_ACCOUNT,
					data: CONTRACT_BYTECODE,
					value: "0x00",
					gasPrice: "0x3B9ACA00",
					gas: "0x100000",
				},
				GENESIS_ACCOUNT_PRIVATE_KEY
			);
	
			const txHash = (await customRequest(context.web3, "eth_sendRawTransaction", [tx.rawTransaction])).result;
	
			// Verify the receipt exists after the block is created
			await createAndFinalizeBlock(context.web3);
			return await context.web3.eth.getTransactionReceipt(txHash);
		}
		
		const receipt = await deployContract();
		expect(receipt.contractAddress).to.equal("0xC2Bf5F29a4384b1aB0C063e1c666f02121B6084a");


		const receipt2 = await deployContract();
		expect(receipt2.contractAddress).to.equal(receipt.contractAddress);
	});
});
