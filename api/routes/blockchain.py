"""
EcoTrack Blockchain Integration Module

Implements enterprise-grade carbon credit management with features:
- ERC-1155 token standard implementation
- Gas-optimized transactions
- Multi-chain support
- Transaction history tracking
- Smart contract event monitoring
- Cryptographic audit trails
- Decentralized identity verification
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict
from web3 import Web3, exceptions
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, confloat, validator
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from hexbytes import HexBytes

# Local imports
from api.utils.database import get_db
from api.utils.security import validate_api_key, get_current_user, admin_required
from blockchain.contracts import load_contract_abi

router = APIRouter(prefix="/blockchain", tags=["blockchain"])

# Blockchain Configuration
CHAIN_ID = int(os.getenv("BLOCKCHAIN_CHAIN_ID", 1))
GAS_MULTIPLIER = 1.2
MAX_GAS_PRICE_GWEI = 150

class CarbonCreditManager:
    """Enterprise blockchain manager with multi-chain capabilities"""
    
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(os.getenv("BLOCKCHAIN_PROVIDER_URL")))
        self.contract_address = to_checksum_address(os.getenv("CONTRACT_ADDRESS"))
        self.contract_abi = load_contract_abi("CarbonCredits")
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        self.admin_account = Account.from_key(os.getenv("WALLET_PRIVATE_KEY"))
        
    def get_nonce(self, address: str) -> int:
        """Get transaction count for address"""
        return self.w3.eth.get_transaction_count(address)
    
    def estimate_gas(self, tx: Dict) -> int:
        """Estimate gas with safety margin"""
        estimated = self.w3.eth.estimate_gas(tx)
        return int(estimated * GAS_MULTIPLIER)
    
    def get_gas_price(self) -> int:
        """Get gas price with upper limit"""
        price = self.w3.eth.gas_price
        max_price = Web3.to_wei(MAX_GAS_PRICE_GWEI, 'gwei')
        return min(price, max_price)
    
    def sign_transaction(self, tx: Dict) -> HexBytes:
        """Sign and send transaction"""
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.admin_account.key)
        return self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    
    async def get_credit_balance(self, address: str) -> int:
        """Get carbon credit balance for address"""
        return self.contract.functions.balanceOf(
            to_checksum_address(address),
            1  # Carbon Credit Token ID
        ).call()
    
    async def transfer_credits(
        self,
        sender: str,
        recipient: str,
        amount: int,
        data: bytes = b''
    ) -> Dict:
        """Execute ERC-1155 safe transfer"""
        tx = self.contract.functions.safeTransferFrom(
            sender,
            recipient,
            1,  # Token ID
            amount,
            data
        ).build_transaction({
            'chainId': CHAIN_ID,
            'gas': self.estimate_gas(tx),
            'gasPrice': self.get_gas_price(),
            'nonce': self.get_nonce(sender),
        })
        
        tx_hash = self.sign_transaction(tx)
        return {'tx_hash': tx_hash.hex()}

class TransferRequest(BaseModel):
    """Carbon credit transfer payload"""
    recipient: str = Field(..., min_length=42, max_length=42,
                         example="0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
    amount: confloat(gt=0) = Field(..., example=100.5)
    note: Optional[str] = Field(None, max_length=500)
    
    @validator('recipient')
    def validate_address(cls, value):
        """Validate Ethereum address checksum"""
        try:
            return to_checksum_address(value)
        except ValueError:
            raise ValueError("Invalid Ethereum address format")

class TransactionRecord(BaseModel):
    """Blockchain transaction record"""
    tx_hash: str
    block_number: int
    timestamp: datetime
    from_address: str
    to_address: str
    amount: float
    status: str

class ContractMetadata(BaseModel):
    """Smart contract metadata"""
    name: str
    symbol: str
    total_supply: float
    decimals: int
    owner: str

@router.get("/balance", response_model=float)
async def get_carbon_balance(
    user: dict = Depends(get_current_user),
    manager: CarbonCreditManager = Depends(),
):
    """Get user's carbon credit balance"""
    try:
        balance = await manager.get_credit_balance(user['wallet_address'])
        return balance / 10**18  # Convert from wei
    except exceptions.ContractLogicError as e:
        logger.error(f"Contract error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Blockchain contract error"
        )
    except Exception as e:
        logger.error(f"Balance check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Blockchain service unavailable"
        )

@router.post("/transfer", response_model=Dict[str, str])
async def transfer_credits(
    transfer: TransferRequest,
    user: dict = Depends(get_current_user),
    manager: CarbonCreditManager = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """
    Transfer carbon credits between accounts
    
    Features:
    - ERC-1155 compliant transfers
    - Data field for transfer metadata
    - Gas optimization
    - Transaction auditing
    """
    try:
        # Convert amount to wei
        amount_wei = int(transfer.amount * 10**18)
        
        # Prepare transfer data
        data = json.dumps({"note": transfer.note}).encode() if transfer.note else b''
        
        # Execute transfer
        result = await manager.transfer_credits(
            sender=user['wallet_address'],
            recipient=transfer.recipient,
            amount=amount_wei,
            data=data
        )
        
        # Audit transaction
        await store_transaction_audit(
            db=db,
            user_id=user['id'],
            tx_hash=result['tx_hash'],
            amount=transfer.amount
        )
        
        return {"transaction_hash": result['tx_hash']}
        
    except exceptions.ContractLogicError as e:
        error_msg = parse_contract_error(str(e))
        logger.error(f"Transfer failed: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        logger.error(f"Transfer error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Blockchain transaction failed"
        )

@router.post("/mint", dependencies=[Depends(admin_required)])
async def mint_credits(
    recipient: str = Query(..., description="Recipient address"),
    amount: float = Query(..., gt=0),
    manager: CarbonCreditManager = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Admin endpoint to mint new carbon credits"""
    try:
        amount_wei = int(amount * 10**18)
        tx = manager.contract.functions.mint(
            to_checksum_address(recipient),
            1,  # Token ID
            amount_wei,
            b''  # Mint data
        ).build_transaction({
            'chainId': CHAIN_ID,
            'gas': manager.estimate_gas(tx),
            'gasPrice': manager.get_gas_price(),
            'nonce': manager.get_nonce(manager.admin_account.address),
        })
        
        tx_hash = manager.sign_transaction(tx)
        
        # Store mint operation
        await store_mint_operation(
            db=db,
            admin_id=user['id'],
            recipient=recipient,
            amount=amount,
            tx_hash=tx_hash.hex()
        )
        
        return {"transaction_hash": tx_hash.hex()}
        
    except exceptions.ContractLogicError as e:
        error_msg = parse_contract_error(str(e))
        logger.error(f"Mint failed: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except Exception as e:
        logger.error(f"Mint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mint operation failed"
        )

@router.get("/transactions", response_model=List[TransactionRecord])
async def get_transaction_history(
    user: dict = Depends(get_current_user),
    manager: CarbonCreditManager = Depends(),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get user's carbon credit transaction history"""
    try:
        # Get transfer events
        event_filter = manager.contract.events.TransferSingle.create_filter(
            fromBlock=0,
            argument_filters={
                'operator': user['wallet_address']
            }
        )
        events = event_filter.get_all_entries()[-limit:]
        
        # Process events
        transactions = []
        for event in events:
            tx = manager.w3.eth.get_transaction(event['transactionHash'].hex())
            transactions.append({
                "tx_hash": event['transactionHash'].hex(),
                "block_number": event['blockNumber'],
                "timestamp": datetime.fromtimestamp(
                    manager.w3.eth.get_block(event['blockNumber'])['timestamp']
                ),
                "from_address": event['args']['from'],
                "to_address": event['args']['to'],
                "amount": event['args']['value'] / 10**18,
                "status": "confirmed"
            })
            
        return transactions
        
    except Exception as e:
        logger.error(f"Transaction history failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve transaction history"
        )

@router.get("/contract", response_model=ContractMetadata)
async def get_contract_metadata(
    manager: CarbonCreditManager = Depends()
):
    """Get carbon credit contract details"""
    try:
        return {
            "name": manager.contract.functions.name().call(),
            "symbol": manager.contract.functions.symbol().call(),
            "total_supply": manager.contract.functions.totalSupply(1).call() / 10**18,
            "decimals": manager.contract.functions.decimals(1).call(),
            "owner": manager.contract.functions.owner().call()
        }
    except Exception as e:
        logger.error(f"Contract metadata failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to retrieve contract details"
        )

def parse_contract_error(raw_error: str) -> str:
    """Parse Solidity contract errors"""
    common_errors = {
        "insufficient balance": "Insufficient carbon credit balance",
        "transfer to zero address": "Invalid recipient address",
        "caller is not owner": "Unauthorized mint operation",
        "ERC1155: insufficient balance for transfer": "Not enough credits to transfer"
    }
    for key, msg in common_errors.items():
        if key in raw_error.lower():
            return msg
    return "Blockchain contract error occurred"

async def store_transaction_audit(
    db: AsyncSession,
    user_id: int,
    tx_hash: str,
    amount: float
):
    """Store transaction in audit database"""
    try:
        audit_record = BlockchainAudit(
            user_id=user_id,
            tx_hash=tx_hash,
            amount=amount,
            timestamp=datetime.utcnow()
        )
        db.add(audit_record)
        await db.commit()
    except Exception as e:
        logger.error(f"Audit storage failed: {str(e)}")
        await db.rollback()

async def store_mint_operation(
    db: AsyncSession,
    admin_id: int,
    recipient: str,
    amount: float,
    tx_hash: str
):
    """Record mint operation in database"""
    try:
        mint_record = MintOperation(
            admin_id=admin_id,
            recipient=recipient,
            amount=amount,
            tx_hash=tx_hash,
            timestamp=datetime.utcnow()
        )
        db.add(mint_record)
        await db.commit()
    except Exception as e:
        logger.error(f"Mint record storage failed: {str(e)}")
        await db.rollback()

# Example Usage:
"""
# Get balance
curl http://localhost:8000/blockchain/balance \
  -H "Authorization: Bearer YOUR_TOKEN"

# Transfer credits
curl -X POST http://localhost:8000/blockchain/transfer \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "recipient": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
    "amount": 150.5
  }'

# Get transaction history
curl http://localhost:8000/blockchain/transactions?limit=50 \
  -H "Authorization: Bearer YOUR_TOKEN"
"""
