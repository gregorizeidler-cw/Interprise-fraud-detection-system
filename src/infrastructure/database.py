"""
Database Management Infrastructure
Handles connections and operations for multiple data sources.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from google.cloud import bigquery
import redis
import os
from contextlib import contextmanager

from ..utils.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Unified database manager for multiple data sources.
    Handles PostgreSQL, BigQuery, and Redis connections.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.engines = {}
        self.bigquery_client = None
        self.redis_client = None
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize all database connections"""
        
        try:
            # PostgreSQL connection for operational data
            self._setup_postgresql()
            
            # BigQuery connection for data lake
            self._setup_bigquery()
            
            # Redis connection for caching
            self._setup_redis()
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database connections: {e}")
            raise
    
    def _setup_postgresql(self):
        """Setup PostgreSQL connection"""
        
        try:
            db_config = self.config.get("data_sources.core_banking")
            
            connection_string = (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False
            )
            
            self.engines['postgresql'] = engine
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Error setting up PostgreSQL: {e}")
            raise
    
    def _setup_bigquery(self):
        """Setup BigQuery connection"""
        
        try:
            bq_config = self.config.get("data_sources.data_lake")
            
            # Initialize BigQuery client
            if bq_config.get('credentials_path'):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = bq_config['credentials_path']
            
            self.bigquery_client = bigquery.Client(
                project=bq_config['project_id']
            )
            
            # Test connection
            query = "SELECT 1 as test"
            self.bigquery_client.query(query).result()
            
            logger.info("BigQuery connection established")
            
        except Exception as e:
            logger.error(f"Error setting up BigQuery: {e}")
            # BigQuery is optional for development
            logger.warning("Continuing without BigQuery connection")
    
    def _setup_redis(self):
        """Setup Redis connection"""
        
        try:
            redis_url = self.config.get("feature_store.online_store.connection_string")
            
            self.redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Error setting up Redis: {e}")
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[List] = None, 
        source: str = "postgresql"
    ) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            source: Data source ('postgresql', 'bigquery')
        
        Returns:
            DataFrame with query results
        """
        
        try:
            if source == "postgresql":
                return self._execute_postgresql_query(query, params)
            elif source == "bigquery":
                return self._execute_bigquery_query(query, params)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def _execute_postgresql_query(
        self, 
        query: str, 
        params: Optional[List] = None
    ) -> pd.DataFrame:
        """Execute PostgreSQL query"""
        
        if 'postgresql' not in self.engines:
            raise RuntimeError("PostgreSQL connection not available")
        
        try:
            with self.engines['postgresql'].connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                return df
                
        except SQLAlchemyError as e:
            logger.error(f"PostgreSQL query error: {e}")
            raise
    
    def _execute_bigquery_query(
        self, 
        query: str, 
        params: Optional[List] = None
    ) -> pd.DataFrame:
        """Execute BigQuery query"""
        
        if not self.bigquery_client:
            raise RuntimeError("BigQuery connection not available")
        
        try:
            # BigQuery doesn't use positional parameters the same way
            # This is a simplified implementation
            job_config = bigquery.QueryJobConfig()
            
            if params:
                # Convert to named parameters (simplified)
                for i, param in enumerate(params):
                    query = query.replace(f"%s", f"'{param}'", 1)
            
            query_job = self.bigquery_client.query(query, job_config=job_config)
            
            # Convert to DataFrame
            df = query_job.to_dataframe()
            
            return df
            
        except Exception as e:
            logger.error(f"BigQuery query error: {e}")
            raise
    
    def insert_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        schema: str = None,
        source: str = "postgresql",
        if_exists: str = "append"
    ) -> bool:
        """
        Insert DataFrame into database table.
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            schema: Database schema
            source: Data source
            if_exists: What to do if table exists ('append', 'replace', 'fail')
        
        Returns:
            True if successful
        """
        
        try:
            if source == "postgresql":
                return self._insert_postgresql_dataframe(df, table_name, schema, if_exists)
            elif source == "bigquery":
                return self._insert_bigquery_dataframe(df, table_name, if_exists)
            else:
                raise ValueError(f"Unsupported data source: {source}")
                
        except Exception as e:
            logger.error(f"Error inserting DataFrame: {e}")
            raise
    
    def _insert_postgresql_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        schema: str = None,
        if_exists: str = "append"
    ) -> bool:
        """Insert DataFrame into PostgreSQL"""
        
        if 'postgresql' not in self.engines:
            raise RuntimeError("PostgreSQL connection not available")
        
        try:
            df.to_sql(
                table_name,
                self.engines['postgresql'],
                schema=schema,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"Inserted {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting into PostgreSQL: {e}")
            raise
    
    def _insert_bigquery_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        if_exists: str = "append"
    ) -> bool:
        """Insert DataFrame into BigQuery"""
        
        if not self.bigquery_client:
            raise RuntimeError("BigQuery connection not available")
        
        try:
            dataset_id = self.config.get("data_sources.data_lake.dataset")
            table_id = f"{dataset_id}.{table_name}"
            
            job_config = bigquery.LoadJobConfig()
            
            if if_exists == "replace":
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            else:
                job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            
            job_config.autodetect = True
            
            job = self.bigquery_client.load_table_from_dataframe(
                df, table_id, job_config=job_config
            )
            
            job.result()  # Wait for the job to complete
            
            logger.info(f"Inserted {len(df)} rows into BigQuery table {table_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting into BigQuery: {e}")
            raise
    
    @contextmanager
    def get_connection(self, source: str = "postgresql"):
        """Get database connection context manager"""
        
        if source == "postgresql":
            if 'postgresql' not in self.engines:
                raise RuntimeError("PostgreSQL connection not available")
            
            with self.engines['postgresql'].connect() as conn:
                yield conn
        else:
            raise ValueError(f"Unsupported source for connection: {source}")
    
    def execute_transaction(
        self, 
        queries: List[str], 
        params_list: Optional[List[List]] = None,
        source: str = "postgresql"
    ) -> bool:
        """
        Execute multiple queries in a transaction.
        
        Args:
            queries: List of SQL queries
            params_list: List of parameter lists for each query
            source: Data source
        
        Returns:
            True if all queries succeeded
        """
        
        if source != "postgresql":
            raise ValueError("Transactions only supported for PostgreSQL")
        
        try:
            with self.get_connection(source) as conn:
                with conn.begin():
                    for i, query in enumerate(queries):
                        params = params_list[i] if params_list else None
                        
                        if params:
                            conn.execute(text(query), params)
                        else:
                            conn.execute(text(query))
            
            logger.info(f"Successfully executed {len(queries)} queries in transaction")
            return True
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    def get_table_info(self, table_name: str, schema: str = None) -> Dict[str, Any]:
        """Get information about a database table"""
        
        try:
            with self.get_connection() as conn:
                metadata = MetaData()
                
                table = Table(
                    table_name, 
                    metadata, 
                    schema=schema, 
                    autoload_with=conn
                )
                
                columns = []
                for column in table.columns:
                    columns.append({
                        'name': column.name,
                        'type': str(column.type),
                        'nullable': column.nullable,
                        'primary_key': column.primary_key
                    })
                
                return {
                    'table_name': table_name,
                    'schema': schema,
                    'columns': columns,
                    'column_count': len(columns)
                }
                
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            raise
    
    def test_connections(self) -> Dict[str, bool]:
        """Test all database connections"""
        
        results = {}
        
        # Test PostgreSQL
        try:
            with self.get_connection("postgresql") as conn:
                conn.execute(text("SELECT 1"))
            results['postgresql'] = True
        except Exception as e:
            logger.error(f"PostgreSQL connection test failed: {e}")
            results['postgresql'] = False
        
        # Test BigQuery
        try:
            if self.bigquery_client:
                query = "SELECT 1 as test"
                self.bigquery_client.query(query).result()
                results['bigquery'] = True
            else:
                results['bigquery'] = False
        except Exception as e:
            logger.error(f"BigQuery connection test failed: {e}")
            results['bigquery'] = False
        
        # Test Redis
        try:
            self.redis_client.ping()
            results['redis'] = True
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            results['redis'] = False
        
        return results
    
    def close_connections(self):
        """Close all database connections"""
        
        try:
            # Close PostgreSQL connections
            for engine in self.engines.values():
                engine.dispose()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    @property
    def engine(self):
        """Get default PostgreSQL engine for backward compatibility"""
        return self.engines.get('postgresql')


class DataWarehouseManager:
    """
    Specialized manager for data warehouse operations.
    Handles large-scale data processing and analytics queries.
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.db_manager = DatabaseManager(config_manager)
    
    def create_feature_tables(self) -> bool:
        """Create feature store tables"""
        
        create_queries = [
            """
            CREATE TABLE IF NOT EXISTS customer_features_profile (
                customer_id VARCHAR(50) PRIMARY KEY,
                customer_age INTEGER,
                account_age_days INTEGER,
                total_products_count INTEGER,
                credit_score_internal INTEGER,
                is_pep BOOLEAN,
                kyc_completion_score FLOAT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS customer_features_behavioral (
                customer_id VARCHAR(50),
                feature_name VARCHAR(100),
                feature_value FLOAT,
                time_window VARCHAR(10),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (customer_id, feature_name, time_window)
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS customer_features_network (
                customer_id VARCHAR(50) PRIMARY KEY,
                customers_sharing_devices INTEGER,
                unique_devices_used INTEGER,
                unique_beneficiaries INTEGER,
                fraudulent_beneficiaries_count INTEGER,
                network_out_degree INTEGER,
                network_in_degree INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                prediction_id SERIAL PRIMARY KEY,
                transaction_id VARCHAR(50),
                customer_id VARCHAR(50),
                product_type VARCHAR(20),
                hub_score FLOAT,
                spoke_score FLOAT,
                final_score FLOAT,
                risk_level VARCHAR(20),
                action VARCHAR(20),
                processing_time_ms FLOAT,
                model_versions JSONB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            """
            CREATE INDEX IF NOT EXISTS idx_fraud_predictions_customer_id 
            ON fraud_predictions (customer_id)
            """,
            
            """
            CREATE INDEX IF NOT EXISTS idx_fraud_predictions_timestamp 
            ON fraud_predictions (timestamp)
            """,
            
            """
            CREATE INDEX IF NOT EXISTS idx_fraud_predictions_risk_level 
            ON fraud_predictions (risk_level)
            """
        ]
        
        try:
            self.db_manager.execute_transaction(create_queries)
            logger.info("Feature store tables created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating feature tables: {e}")
            return False
    
    def optimize_tables(self) -> bool:
        """Optimize database tables for performance"""
        
        optimization_queries = [
            "VACUUM ANALYZE customer_features_profile",
            "VACUUM ANALYZE customer_features_behavioral", 
            "VACUUM ANALYZE customer_features_network",
            "VACUUM ANALYZE fraud_predictions",
            "REINDEX TABLE fraud_predictions"
        ]
        
        try:
            for query in optimization_queries:
                self.db_manager.execute_query(query)
            
            logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing tables: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        stats_query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples,
                last_vacuum,
                last_analyze
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY n_live_tup DESC
        """
        
        try:
            stats_df = self.db_manager.execute_query(stats_query)
            
            return {
                'table_stats': stats_df.to_dict('records'),
                'total_tables': len(stats_df),
                'total_live_tuples': stats_df['live_tuples'].sum(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}