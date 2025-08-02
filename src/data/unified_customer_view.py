"""
Unified Customer View - 360° Customer Profile System
Consolidates data from multiple sources and performs entity resolution.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass
import hashlib
import json

from .schemas import CustomerProfile, CustomerIdentity, AccountProduct, Transaction
from ..infrastructure.database import DatabaseManager
from ..utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


@dataclass
class EntityResolutionRule:
    """Rule for entity resolution"""
    name: str
    fields: List[str]
    weight: float
    exact_match: bool = True
    similarity_threshold: float = 0.9


class EntityResolver:
    """
    Advanced entity resolution for customer identity matching.
    Handles fuzzy matching and probabilistic record linkage.
    """
    
    def __init__(self):
        self.resolution_rules = [
            EntityResolutionRule("cpf_exact", ["document_number"], 1.0, True),
            EntityResolutionRule("email_exact", ["email"], 0.8, True),
            EntityResolutionRule("phone_exact", ["phone"], 0.7, True),
            EntityResolutionRule("name_birth", ["full_name", "birth_date"], 0.9, False, 0.85),
            EntityResolutionRule("name_phone", ["full_name", "phone"], 0.8, False, 0.80),
        ]
    
    def calculate_similarity(self, record1: Dict, record2: Dict) -> float:
        """Calculate similarity score between two customer records"""
        total_score = 0.0
        total_weight = 0.0
        
        for rule in self.resolution_rules:
            if all(field in record1 and field in record2 for field in rule.fields):
                if rule.exact_match:
                    match = all(
                        record1[field] == record2[field] 
                        for field in rule.fields
                        if record1[field] and record2[field]
                    )
                    score = 1.0 if match else 0.0
                else:
                    score = self._fuzzy_match(record1, record2, rule.fields)
                
                if score >= rule.similarity_threshold:
                    total_score += score * rule.weight
                
                total_weight += rule.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _fuzzy_match(self, record1: Dict, record2: Dict, fields: List[str]) -> float:
        """Perform fuzzy matching on specified fields"""
        from difflib import SequenceMatcher
        
        scores = []
        for field in fields:
            val1 = str(record1.get(field, "")).lower().strip()
            val2 = str(record2.get(field, "")).lower().strip()
            
            if val1 and val2:
                similarity = SequenceMatcher(None, val1, val2).ratio()
                scores.append(similarity)
        
        return np.mean(scores) if scores else 0.0
    
    def resolve_entities(self, records: List[Dict], threshold: float = 0.8) -> Dict[str, List[str]]:
        """
        Resolve entities and return groups of matching customer IDs.
        Returns dict where key is canonical customer_id and value is list of all matching IDs.
        """
        groups = {}
        processed = set()
        
        for i, record1 in enumerate(records):
            if record1['customer_id'] in processed:
                continue
                
            group = [record1['customer_id']]
            processed.add(record1['customer_id'])
            
            for j, record2 in enumerate(records[i+1:], i+1):
                if record2['customer_id'] in processed:
                    continue
                    
                similarity = self.calculate_similarity(record1, record2)
                if similarity >= threshold:
                    group.append(record2['customer_id'])
                    processed.add(record2['customer_id'])
            
            # Use the oldest customer_id as canonical
            canonical_id = min(group)
            groups[canonical_id] = group
        
        return groups


class CustomerDataConsolidator:
    """
    Consolidates customer data from multiple sources into unified customer profiles.
    """
    
    def __init__(self, config_manager: ConfigManager, db_manager: DatabaseManager):
        self.config = config_manager
        self.db = db_manager
        self.entity_resolver = EntityResolver()
    
    def extract_customer_identities(self) -> pd.DataFrame:
        """Extract customer identity data from all source systems"""
        
        queries = {
            'core_banking': """
                SELECT 
                    customer_id,
                    document_number,
                    full_name,
                    birth_date,
                    email,
                    phone,
                    created_at,
                    updated_at,
                    'core_banking' as source_system
                FROM customers
                WHERE status = 'active'
            """,
            
            'credit_cards': """
                SELECT 
                    customer_id,
                    document_number,
                    cardholder_name as full_name,
                    birth_date,
                    email,
                    phone,
                    account_opened as created_at,
                    last_updated as updated_at,
                    'credit_cards' as source_system
                FROM card_customers
                WHERE card_status IN ('active', 'suspended')
            """,
            
            'loans': """
                SELECT 
                    borrower_id as customer_id,
                    cpf as document_number,
                    borrower_name as full_name,
                    birth_date,
                    email,
                    contact_phone as phone,
                    application_date as created_at,
                    last_modified as updated_at,
                    'loans' as source_system
                FROM loan_applications
                WHERE status != 'cancelled'
            """
        }
        
        all_identities = []
        
        for source, query in queries.items():
            try:
                df = self.db.execute_query(query, source)
                all_identities.append(df)
                logger.info(f"Extracted {len(df)} identities from {source}")
            except Exception as e:
                logger.error(f"Failed to extract from {source}: {e}")
        
        if all_identities:
            combined_df = pd.concat(all_identities, ignore_index=True)
            # Clean and standardize data
            combined_df = self._clean_identity_data(combined_df)
            return combined_df
        else:
            return pd.DataFrame()
    
    def _clean_identity_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize customer identity data"""
        
        # Clean document numbers (remove non-digits)
        df['document_number'] = df['document_number'].str.replace(r'\D', '', regex=True)
        
        # Standardize names
        df['full_name'] = (df['full_name']
                          .str.upper()
                          .str.strip()
                          .str.replace(r'\s+', ' ', regex=True))
        
        # Clean phone numbers
        df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)
        
        # Clean emails
        df['email'] = df['email'].str.lower().str.strip()
        
        # Remove duplicates within same source
        df = df.drop_duplicates(subset=['customer_id', 'source_system'])
        
        return df
    
    def perform_entity_resolution(self, identities_df: pd.DataFrame) -> Dict[str, str]:
        """
        Perform entity resolution and return mapping from customer_id to canonical_customer_id
        """
        
        # Convert to records for entity resolution
        records = identities_df.to_dict('records')
        
        # Perform entity resolution
        entity_groups = self.entity_resolver.resolve_entities(records)
        
        # Create mapping from customer_id to canonical_customer_id
        id_mapping = {}
        for canonical_id, group_ids in entity_groups.items():
            for customer_id in group_ids:
                id_mapping[customer_id] = canonical_id
        
        logger.info(f"Entity resolution: {len(records)} records -> {len(entity_groups)} unique entities")
        
        return id_mapping
    
    def build_customer_profiles(self, canonical_mapping: Dict[str, str]) -> List[CustomerProfile]:
        """Build unified customer profiles using canonical customer IDs"""
        
        profiles = []
        
        for canonical_id in set(canonical_mapping.values()):
            # Get all customer IDs that map to this canonical ID
            related_ids = [k for k, v in canonical_mapping.items() if v == canonical_id]
            
            try:
                profile = self._build_single_profile(canonical_id, related_ids)
                profiles.append(profile)
            except Exception as e:
                logger.error(f"Failed to build profile for {canonical_id}: {e}")
        
        return profiles
    
    def _build_single_profile(self, canonical_id: str, related_ids: List[str]) -> CustomerProfile:
        """Build a single unified customer profile"""
        
        # Get the best identity information
        identity = self._get_best_identity(related_ids)
        
        # Get all products/accounts
        products = self._get_customer_products(related_ids)
        
        # Calculate relationship metrics
        relationship_metrics = self._calculate_relationship_metrics(related_ids)
        
        # Get risk indicators
        risk_indicators = self._get_risk_indicators(related_ids)
        
        # Build the profile
        profile = CustomerProfile(
            customer_id=canonical_id,
            identity=identity,
            
            # Demographics
            age=self._calculate_age(identity.birth_date) if identity.birth_date else None,
            
            # Banking relationship
            customer_since=relationship_metrics['customer_since'],
            relationship_length_days=relationship_metrics['relationship_length_days'],
            total_products=len(products),
            active_products=[p.product_type for p in products if p.status == 'active'],
            primary_product=self._determine_primary_product(products),
            
            # Risk indicators
            is_pep=risk_indicators.get('is_pep', False),
            credit_score_internal=risk_indicators.get('credit_score_internal'),
            credit_score_external=risk_indicators.get('credit_score_external'),
            has_fraud_history=risk_indicators.get('has_fraud_history', False),
            
            # Will be calculated separately
            last_activity=self._get_last_activity(related_ids)
        )
        
        return profile
    
    def _get_best_identity(self, customer_ids: List[str]) -> CustomerIdentity:
        """Get the most complete and recent identity information"""
        
        query = """
            SELECT 
                customer_id,
                document_number,
                full_name,
                birth_date,
                email,
                phone,
                created_at,
                updated_at,
                source_system
            FROM unified_customer_identities
            WHERE customer_id IN ({})
            ORDER BY 
                CASE WHEN email IS NOT NULL THEN 1 ELSE 0 END DESC,
                CASE WHEN phone IS NOT NULL THEN 1 ELSE 0 END DESC,
                updated_at DESC
            LIMIT 1
        """.format(','.join([f"'{id}'" for id in customer_ids]))
        
        result = self.db.execute_query(query).iloc[0]
        
        return CustomerIdentity(
            customer_id=customer_ids[0],  # Use first ID as canonical
            document_number=result['document_number'],
            full_name=result['full_name'],
            birth_date=result['birth_date'],
            email=result['email'],
            phone=result['phone'],
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
    
    def _get_customer_products(self, customer_ids: List[str]) -> List[AccountProduct]:
        """Get all products for customer IDs"""
        
        query = """
            SELECT 
                account_id,
                customer_id,
                product_type,
                account_number,
                status,
                balance,
                credit_limit,
                opened_date,
                closed_date,
                last_activity
            FROM unified_customer_products
            WHERE customer_id IN ({})
        """.format(','.join([f"'{id}'" for id in customer_ids]))
        
        results = self.db.execute_query(query)
        
        products = []
        for _, row in results.iterrows():
            product = AccountProduct(
                account_id=row['account_id'],
                customer_id=row['customer_id'],
                product_type=row['product_type'],
                account_number=row['account_number'],
                status=row['status'],
                balance=row['balance'],
                credit_limit=row['credit_limit'],
                opened_date=row['opened_date'],
                closed_date=row['closed_date'],
                last_activity=row['last_activity'],
                days_since_opening=(datetime.now() - row['opened_date']).days
            )
            products.append(product)
        
        return products
    
    def _calculate_relationship_metrics(self, customer_ids: List[str]) -> Dict:
        """Calculate customer relationship metrics"""
        
        query = """
            SELECT 
                MIN(created_at) as first_account,
                COUNT(*) as total_accounts
            FROM unified_customer_products
            WHERE customer_id IN ({})
        """.format(','.join([f"'{id}'" for id in customer_ids]))
        
        result = self.db.execute_query(query).iloc[0]
        
        customer_since = result['first_account']
        relationship_length = (datetime.now() - customer_since).days
        
        return {
            'customer_since': customer_since,
            'relationship_length_days': relationship_length
        }
    
    def _get_risk_indicators(self, customer_ids: List[str]) -> Dict:
        """Get risk indicators for customer"""
        
        query = """
            SELECT 
                MAX(CASE WHEN ri.indicator_type = 'pep' THEN 1 ELSE 0 END) as is_pep,
                MAX(CASE WHEN ri.indicator_type = 'fraud_history' THEN 1 ELSE 0 END) as has_fraud_history,
                AVG(CASE WHEN ri.indicator_type = 'credit_score_internal' THEN ri.indicator_value END) as credit_score_internal,
                AVG(CASE WHEN ri.indicator_type = 'credit_score_external' THEN ri.indicator_value END) as credit_score_external
            FROM customer_risk_indicators ri
            WHERE ri.customer_id IN ({})
        """.format(','.join([f"'{id}'" for id in customer_ids]))
        
        result = self.db.execute_query(query)
        
        if len(result) > 0:
            row = result.iloc[0]
            return {
                'is_pep': bool(row['is_pep']),
                'has_fraud_history': bool(row['has_fraud_history']),
                'credit_score_internal': int(row['credit_score_internal']) if pd.notna(row['credit_score_internal']) else None,
                'credit_score_external': int(row['credit_score_external']) if pd.notna(row['credit_score_external']) else None
            }
        
        return {}
    
    def _determine_primary_product(self, products: List[AccountProduct]) -> Optional[str]:
        """Determine the primary product based on activity and balance"""
        
        if not products:
            return None
        
        # Priority: active products with recent activity and high balance
        active_products = [p for p in products if p.status == 'active']
        
        if not active_products:
            return None
        
        # Sort by last activity and balance
        primary = max(active_products, 
                     key=lambda p: (
                         p.last_activity or datetime.min,
                         p.balance or 0
                     ))
        
        return primary.product_type
    
    def _get_last_activity(self, customer_ids: List[str]) -> Optional[datetime]:
        """Get the most recent activity across all products"""
        
        query = """
            SELECT MAX(last_activity) as last_activity
            FROM unified_customer_products
            WHERE customer_id IN ({})
        """.format(','.join([f"'{id}'" for id in customer_ids]))
        
        result = self.db.execute_query(query)
        
        if len(result) > 0 and pd.notna(result.iloc[0]['last_activity']):
            return result.iloc[0]['last_activity']
        
        return None
    
    def _calculate_age(self, birth_date: datetime) -> int:
        """Calculate age from birth date"""
        if birth_date:
            today = datetime.now()
            return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return None


class UnifiedCustomerViewManager:
    """
    Main manager for the Unified Customer View system.
    Orchestrates data extraction, entity resolution, and profile building.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db = DatabaseManager(config_manager)
        self.consolidator = CustomerDataConsolidator(config_manager, self.db)
    
    def build_unified_view(self) -> List[CustomerProfile]:
        """Build complete unified customer view"""
        
        logger.info("Starting unified customer view build process")
        
        # Step 1: Extract customer identities from all sources
        logger.info("Extracting customer identities...")
        identities_df = self.consolidator.extract_customer_identities()
        
        if identities_df.empty:
            logger.warning("No customer identities found")
            return []
        
        # Step 2: Perform entity resolution
        logger.info("Performing entity resolution...")
        canonical_mapping = self.consolidator.perform_entity_resolution(identities_df)
        
        # Step 3: Build unified customer profiles
        logger.info("Building unified customer profiles...")
        profiles = self.consolidator.build_customer_profiles(canonical_mapping)
        
        logger.info(f"Successfully built {len(profiles)} unified customer profiles")
        
        return profiles
    
    def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Get unified profile for a specific customer"""
        
        # First check if this customer_id is canonical or needs mapping
        mapping_query = """
            SELECT canonical_customer_id 
            FROM customer_id_mapping 
            WHERE customer_id = %s
        """
        
        result = self.db.execute_query(mapping_query, params=[customer_id])
        
        if len(result) > 0:
            canonical_id = result.iloc[0]['canonical_customer_id']
        else:
            canonical_id = customer_id
        
        # Get the unified profile
        profile_query = """
            SELECT * FROM unified_customer_profiles 
            WHERE customer_id = %s
        """
        
        result = self.db.execute_query(profile_query, params=[canonical_id])
        
        if len(result) > 0:
            # Convert to CustomerProfile object
            # This would need proper deserialization
            return CustomerProfile(**result.iloc[0].to_dict())
        
        return None
    
    def update_customer_profile(self, customer_id: str) -> bool:
        """Update a specific customer's unified profile"""
        
        try:
            # Get all related customer IDs
            related_ids = self._get_related_customer_ids(customer_id)
            
            # Rebuild profile for this customer
            profile = self.consolidator._build_single_profile(customer_id, related_ids)
            
            # Save to database
            self._save_customer_profile(profile)
            
            logger.info(f"Updated profile for customer {customer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update profile for {customer_id}: {e}")
            return False
    
    def _get_related_customer_ids(self, customer_id: str) -> List[str]:
        """Get all customer IDs related to a canonical customer ID"""
        
        query = """
            SELECT customer_id 
            FROM customer_id_mapping 
            WHERE canonical_customer_id = (
                SELECT canonical_customer_id 
                FROM customer_id_mapping 
                WHERE customer_id = %s
            )
        """
        
        result = self.db.execute_query(query, params=[customer_id])
        return result['customer_id'].tolist()
    
    def _save_customer_profile(self, profile: CustomerProfile):
        """Save customer profile to database"""
        
        # Convert profile to database format and save
        # This would involve proper serialization and database operations
        pass