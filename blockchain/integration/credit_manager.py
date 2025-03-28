import json
import os
from web3 import Web3, HTTPProvider
from web3.middleware import geth_poa_middleware

class CarbonCreditManager:
    def __init__(self, network_url=None, contract_address=None):
        self.w3 = Web3(HTTPProvider(
            network_url or os.getenv("ETH_NETWORK_URL", "http://localhost:8545")
        ))
        
        # Add POA middleware for networks like Polygon
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        
        self.contract_address = contract_address or os.getenv("CONTRACT_ADDRESS")
        self.abi = self._load_abi()
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.abi
        )
    
    def _load_abi(self):
        with open("blockchain/build/contracts/CarbonCredits.json") as f:
            contract_json = json.load(f)
            return contract_json["abi"]
    
    def get_balance(self, address):
        """Get carbon credit balance for an address"""
        return self.contract.functions.balanceOf(address).call()
    
    def transfer_credits(self, sender_privkey, recipient, amount):
        """Transfer credits between accounts"""
        sender = self.w3.eth.account.privateKeyToAccount(sender_privkey).address
        nonce = self.w3.eth.getTransactionCount(sender)
        
        txn = self.contract.functions.transfer(
            recipient,
            amount
        ).buildTransaction({
            'chainId': self.w3.eth.chain_id,
            'gas': 200000,
            'gasPrice': self.w3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(txn, sender_privkey)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        return tx_hash.hex()
    
    def mint_credits(self, minter_privkey, recipient, amount):
        """Mint new carbon credits (requires MINTER_ROLE)"""
        minter = self.w3.eth.account.privateKeyToAccount(minter_privkey).address
        nonce = self.w3.eth.getTransactionCount(minter)
        
        txn = self.contract.functions.mint(
            recipient,
            amount
        ).buildTransaction({
            'chainId': self.w3.eth.chain_id,
            'gas': 300000,
            'gasPrice': self.w3.toWei('50', 'gwei'),
            'nonce': nonce,
        })
        
        signed_txn = self.w3.eth.account.sign_transaction(txn, minter_privkey)
        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        return tx_hash.hex()
    
    def get_transaction_history(self, address, num_blocks=1000):
        """Get recent credit transactions for an address"""
        latest_block = self.w3.eth.block_number
        events = self.contract.events.CreditsTransferred.getLogs(
            fromBlock=latest_block - num_blocks,
            toBlock='latest',
            argument_filters={'from': address}
        )
        return events
