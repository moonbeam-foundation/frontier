import { expect, use as chaiUse } from "chai";
import chaiAsPromised from "chai-as-promised";
import Web3 from "web3";
import { Contract } from "web3-eth-contract";
import { AbiItem } from "web3-utils";

import Test from "../build/contracts/Test.json";
import { GENESIS_ACCOUNT, GENESIS_ACCOUNT_PRIVATE_KEY } from "./config";
import { createAndFinalizeBlock, customRequest, describeWithFrontier } from "./util";

chaiUse(chaiAsPromised);

async function deployContract(
	web3: Web3,
	bytecode: string,
	abi: AbiItem[]
): Promise<{
	contract: Contract;
	contractAddress: string;
}> {
	const contract = new web3.eth.Contract(abi);
	const data = contract.deploy({ data: bytecode }).encodeABI();
	const tx = await web3.eth.accounts.signTransaction(
		{
			from: GENESIS_ACCOUNT,
			data,
			value: "0x00",
			gas: "0x100000",
		},
		GENESIS_ACCOUNT_PRIVATE_KEY
	);
	const { result } = await customRequest(web3, "eth_sendRawTransaction", [tx.rawTransaction]);
	await createAndFinalizeBlock(web3);
	const receipt = await web3.eth.getTransactionReceipt(result);
	const contractAddress = receipt.contractAddress;

	return { contract, contractAddress };
}

describeWithFrontier("Frontier RPC (EthCall)", (context) => {
	let contract: Contract;
	let contractAddress: string;
	let contractAddress2: string;

	before("deploy contract to two addresses", async function () {
		this.timeout(15000);
		const firstDeploy = await deployContract(context.web3, Test.bytecode, Test.abi as AbiItem[]);
		const secondDeploy = await deployContract(context.web3, Test.bytecode, Test.abi as AbiItem[]);
		contract = firstDeploy.contract;
		contractAddress = firstDeploy.contractAddress;
		contractAddress2 = secondDeploy.contractAddress;
	});

	it("should be able to call eth_call from address with no code", async function () {
		const r = await customRequest(context.web3, "eth_call", [
			{
				from: GENESIS_ACCOUNT,
				to: contractAddress,
				data: await contract.methods.multiply(5).encodeABI(),
			},
		]);
		expect(r.error?.message).to.be.undefined;
		expect(Web3.utils.hexToNumberString(r.result)).to.equal("35");
	});

	it("should be able to call eth_call from address with code", async function () {
		const r = await customRequest(context.web3, "eth_call", [
			{
				from: contractAddress2,
				to: contractAddress,
				data: await contract.methods.multiply(5).encodeABI(),
			},
		]);
		expect(r.error?.message).to.be.undefined;
		expect(Web3.utils.hexToNumberString(r.result)).to.equal("35");
	});
});
