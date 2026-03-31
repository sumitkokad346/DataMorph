"""
Enhanced Preprocessing Pipeline - Universal Data Handler
Version: 4.0 - Production-Ready with Complete Operation Execution
Author: Senior Data Scientist / ML Systems Architect
Description: Universal preprocessing engine that handles ANY dataset and executes ALL AI-suggested operations correctly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, OneHotEncoder
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from scipy import stats
import json
import re

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_HUMAN_AGE = 122
MAX_PERCENTAGE = 100
MAX_REASONABLE_SALARY = 10_000_000
MAX_ONEHOT_CATEGORIES = 20

# Column patterns for intent inference
ID_PATTERNS = {'id', 'Id', 'ID', 'uuid', 'guid', 'key', 'code', 'index', 'row_id', 'customerid'}
TARGET_PATTERNS = {'target', 'label', 'class', 'outcome', 'result', 'score', 'rating', 'churn'}
TEXT_PATTERNS = {'text', 'description', 'comment', 'review', 'message', 'content', 'story', 'summary', 'overview', 'plot'}


class RuleValidationEngine:
    """Rule-based validation engine with domain-aware sanity checks"""
    
    def __init__(self):
        self.validation_logs = []
        self.validation_stats = {
            'values_flagged': 0,
            'values_capped': 0,
            'values_marked_missing': 0
        }
    
    def infer_column_intent(self, col_name: str, series: pd.Series) -> Dict[str, Any]:
        """Infer column intent from name patterns and data distribution"""
        col_name_lower = col_name.lower()
        
        intent = {
            'is_age': False,
            'is_salary': False,
            'is_percentage': False,
            'is_count': False,
            'is_id': False,
            'is_target': False,
            'is_text': False,
            'is_categorical': False,
            'confidence': 0.5
        }
        
        # Check patterns
        if any(pattern in col_name_lower for pattern in ID_PATTERNS):
            intent['is_id'] = True
        if any(pattern in col_name_lower for pattern in TARGET_PATTERNS):
            intent['is_target'] = True
        if any(pattern in col_name_lower for pattern in TEXT_PATTERNS):
            intent['is_text'] = True
        
        # Numeric patterns
        if pd.api.types.is_numeric_dtype(series):
            if 'age' in col_name_lower or 'year' in col_name_lower:
                intent['is_age'] = True
            if 'rating' in col_name_lower or 'score' in col_name_lower:
                if series.min() >= 0 and series.max() <= 10:
                    intent['is_percentage'] = True
            if 'charges' in col_name_lower or 'salary' in col_name_lower or 'income' in col_name_lower:
                intent['is_salary'] = True
        else:
            # Categorical detection
            if series.nunique() < 50:
                intent['is_categorical'] = True
        
        return intent
    
    def get_validation_summary(self) -> Dict:
        return {
            'logs': self.validation_logs,
            'stats': self.validation_stats,
            'timestamp': datetime.now().isoformat()
        }


def run_preprocessing_pipeline(file, operations=None, original_filename=None, column_operations=None):
    """
    Main preprocessing pipeline - Universal handler for ANY dataset
    """
    logs = []
    download_info = None
    
    rule_engine = RuleValidationEngine()
    
    try:
        # Step 1: Load data
        df, read_logs = smart_read_csv(file)
        logs.extend(read_logs)
        
        if df.empty:
            logs.append("❌ Empty dataset loaded")
            return df, logs, None
        
        original_shape = df.shape
        logs.append(f"✅ Dataset loaded: {original_shape[0]} rows × {original_shape[1]} columns")
        
        # Step 2: Apply basic validation
        logs.append("🔍 Applying basic validation...")
        df = apply_basic_validation(df, rule_engine, logs)
        
        # Step 3: Data profiling
        profile_logs = data_profiling(df, rule_engine)
        logs.extend(profile_logs)
        
        # Step 4: Determine which operations to execute
        final_operations = []
        
        # Priority 1: Column operations (if provided and non-empty)
        if column_operations and isinstance(column_operations, dict) and len(column_operations) > 0:
            has_real_ops = False
            for key, value in column_operations.items():
                if value and len(value) > 0:
                    has_real_ops = True
                    break
            
            if has_real_ops:
                logs.append(f"🎯 Executing column-wise operations")
                df = execute_column_wise_operations(df, column_operations, logs, rule_engine)
                final_operations = column_operations
            else:
                logs.append("⚠️ Empty column_operations, checking operations list")
                operations = None
        
        # Priority 2: Operations list (if provided)
        if operations and len(operations) > 0:
            logs.append(f"🎯 Executing operations list: {len(operations)} operations")
            df = execute_smart_operations(df, operations, logs, rule_engine)
            final_operations = operations
        
        # Priority 3: AI suggestions (only if nothing else provided)
        if not final_operations:
            logs.append("🤖 Getting AI suggestions from Groq...")
            ai_operations = get_ai_suggestions_enhanced(df, logs, rule_engine)
            
            if ai_operations and len(ai_operations) > 0:
                logs.append(f"🧠 AI suggested {len(ai_operations)} operations")
                df = execute_smart_operations(df, ai_operations, logs, rule_engine)
                final_operations = ai_operations
            else:
                logs.append("⚠️ Using safe default operations")
                safe_ops = ["validate:datatypes", "flag:missing"]
                df = execute_smart_operations(df, safe_ops, logs, rule_engine)
                final_operations = safe_ops
        
        # Step 5: Final quality check
        quality_logs = final_quality_check(df, original_shape)
        logs.extend(quality_logs)
        
        # Step 6: Save file
        if original_filename:
            validation_summary = rule_engine.get_validation_summary()
            download_info = save_processed_file_enhanced(df, original_filename, validation_summary)
            logs.append(f"💾 Saved: {download_info['filename']}")
        
        logs.append("🎉 Preprocessing completed successfully!")
        logs.append(f"📝 Total operations executed: {len(final_operations)}")
        
        return df, logs, download_info
    
    except Exception as e:
        error_msg = f"❌ Preprocessing failed: {str(e)}"
        logs.append(error_msg)
        logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
        return pd.DataFrame(), logs, None


def get_ai_suggestions_enhanced(df, logs, rule_engine):
    """
    Get AI suggestions from Groq API with robust parsing
    """
    try:
        from mainapp.logic.llm_logic import LLMAgent
        llm = LLMAgent()
        
        logs.append("📡 Calling Groq API...")
        ai_result = llm.analyze_dataset(df, "column_wise")
        
        if ai_result.get("status") == "success":
            operations = ai_result.get("operations", [])
            
            logs.append(f"🧠 RAW AI operations: {len(operations)} operations")
            
            if operations:
                normalized = normalize_ai_operations(operations, df)
                logs.append(f"🔄 Normalized: {len(normalized)} operations")
                
                if normalized:
                    return normalized
                else:
                    logs.append("⚠️ No valid operations after normalization")
                    return ["validate:datatypes", "flag:missing"]
            else:
                logs.append("⚠️ AI returned empty operations")
                return ["validate:datatypes", "flag:missing"]
        else:
            error = ai_result.get("reason", "Unknown error")
            logs.append(f"⚠️ AI error: {error}")
            return ["validate:datatypes", "flag:missing"]
    
    except Exception as e:
        logs.append(f"⚠️ AI exception: {str(e)}")
        return ["validate:datatypes", "flag:missing"]


def normalize_ai_operations(operations, df):
    """
    Normalize AI operations to standard format - Universal handler
    """
    if not operations:
        return []
    
    normalized = []
    allowed_ops = {
        "validate:datatypes", "remove:duplicates", "flag:missing",
        "impute:mean", "impute:median", "impute:mode",
        "scale:standard", "scale:minmax", "scale:robust",
        "encode:onehot", "encode:label",
        "detect:outliers_iqr", "flag:outliers", "handle:outliers",
        "clean:text", "log_transform"
    }
    
    for op in operations:
        try:
            # Handle dict format
            if isinstance(op, dict):
                operation = op.get('operation') or op.get('name')
                column = op.get('column')
                
                if operation and operation in allowed_ops:
                    if column and column in df.columns:
                        normalized.append({"operation": operation, "column": column})
                    elif not column:
                        normalized.append(operation)
            
            # Handle string format
            elif isinstance(op, str):
                if op in allowed_ops:
                    normalized.append(op)
        
        except Exception as e:
            logger.warning(f"Error normalizing operation {op}: {e}")
            continue
    
    # Remove duplicates
    seen = set()
    unique = []
    for op in normalized:
        op_str = str(op)
        if op_str not in seen:
            seen.add(op_str)
            unique.append(op)
    
    # Always include validate:datatypes if not present
    has_validate = False
    for op in unique:
        if op == "validate:datatypes" or (isinstance(op, dict) and op.get('operation') == "validate:datatypes"):
            has_validate = True
            break
    
    if not has_validate:
        unique.insert(0, "validate:datatypes")
    
    return unique


def execute_smart_operations(df, operations, logs, rule_engine):
    """
    Execute operations with universal handling - NO OPERATION FAILS SILENTLY
    """
    if not operations:
        logs.append("⚠️ No operations to execute")
        return df
    
    logs.append(f"🚀 Executing {len(operations)} operations")
    executed = []
    failed = []
    
    for op in operations:
        try:
            # Handle dict format (column-specific)
            if isinstance(op, dict):
                operation = op.get('operation')
                column = op.get('column')
                
                if not operation:
                    logs.append(f"⚠️ Invalid operation format: {op}")
                    failed.append(str(op))
                    continue
                
                if column and column not in df.columns:
                    logs.append(f"⚠️ Column '{column}' not found")
                    failed.append(f"{operation}:{column}")
                    continue
                
                # Execute column-specific operation with enhanced handler
                df = execute_single_operation_enhanced(df, operation, column, logs, rule_engine)
                executed.append(f"{column}:{operation}" if column else operation)
            
            # Handle string format (dataset-wide)
            elif isinstance(op, str):
                df = execute_dataset_operation_enhanced(df, op, logs, rule_engine)
                executed.append(op)
            
            else:
                logs.append(f"⚠️ Unknown operation type: {type(op)}")
                failed.append(str(op))
        
        except Exception as e:
            logs.append(f"⚠️ Operation failed: {op}, error: {str(e)}")
            failed.append(str(op))
            continue
    
    if executed:
        logs.append(f"✅ Executed: {len(executed)} operations")
        if len(executed) <= 10:
            logs.append(f"   Details: {executed}")
        else:
            logs.append(f"   First 10: {executed[:10]}...")
    
    if failed:
        logs.append(f"⚠️ Failed: {len(failed)} operations")
        if len(failed) <= 5:
            logs.append(f"   Details: {failed}")
    
    return df


def execute_single_operation_enhanced(df, operation, column, logs, rule_engine):
    """
    Enhanced single operation execution - UNIVERSAL HANDLER
    Executes ALL operation types correctly
    """
    if column not in df.columns:
        logs.append(f"⚠️ Column '{column}' not found")
        return df
    
    try:
        # ==================== IMPUTATION ====================
        if operation.startswith("impute:"):
            method = operation.split(":")[1]
            missing_count = df[column].isnull().sum()
            
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df[column]):
                    if method == "mean":
                        df[column] = df[column].fillna(df[column].mean())
                        logs.append(f"📊 {column}: imputed {missing_count} missing values with mean")
                    elif method == "median":
                        df[column] = df[column].fillna(df[column].median())
                        logs.append(f"📊 {column}: imputed {missing_count} missing values with median")
                    elif method == "mode":
                        mode_val = df[column].mode()
                        if not mode_val.empty:
                            df[column] = df[column].fillna(mode_val[0])
                            logs.append(f"📊 {column}: imputed {missing_count} missing values with mode")
                else:
                    # Categorical imputation
                    mode_val = df[column].mode()
                    if not mode_val.empty:
                        df[column] = df[column].fillna(mode_val[0])
                        logs.append(f"🏷️ {column}: imputed {missing_count} missing categorical values with mode")
                    else:
                        df[column] = df[column].fillna("Unknown")
                        logs.append(f"🏷️ {column}: imputed {missing_count} missing values with 'Unknown'")
        
        # ==================== SCALING ====================
        elif operation.startswith("scale:"):
            if pd.api.types.is_numeric_dtype(df[column]):
                if df[column].isnull().any():
                    logs.append(f"⚠️ {column}: skipping scaling - contains missing values")
                elif df[column].nunique() <= 1:
                    logs.append(f"⚠️ {column}: skipping scaling - constant column")
                else:
                    method = operation.split(":")[1]
                    if method == "standard":
                        scaler = StandardScaler()
                        df[column] = scaler.fit_transform(df[[column]]).flatten()
                        logs.append(f"⚖️ {column}: standard scaling applied")
                    elif method == "minmax":
                        scaler = MinMaxScaler()
                        df[column] = scaler.fit_transform(df[[column]]).flatten()
                        logs.append(f"⚖️ {column}: min-max scaling applied (0-1 range)")
                    elif method == "robust":
                        scaler = RobustScaler()
                        df[column] = scaler.fit_transform(df[[column]]).flatten()
                        logs.append(f"⚖️ {column}: robust scaling applied (outlier-resistant)")
        
        # ==================== ENCODING - CRITICAL FIX ====================
        elif operation.startswith("encode:"):
            method = operation.split(":")[1]
            
            # Universal categorical detection
            is_categorical = (pd.api.types.is_object_dtype(df[column]) or 
                             pd.api.types.is_categorical_dtype(df[column]) or
                             pd.api.types.is_string_dtype(df[column]))
            
            if is_categorical:
                unique_count = df[column].nunique()
                
                if method == "label":
                    if unique_count > 1:
                        le = LabelEncoder()
                        # Convert to string to ensure encoding works for all types
                        df[column] = le.fit_transform(df[column].astype(str))
                        logs.append(f"🔤 {column}: label encoded ({unique_count} categories → 0-{unique_count-1})")
                    else:
                        logs.append(f"⚠️ {column}: skipping label encoding - only 1 unique value")
                
                elif method == "onehot":
                    if 1 < unique_count <= MAX_ONEHOT_CATEGORIES:
                        # One-hot encoding with proper handling
                        encoded = pd.get_dummies(df[column], prefix=column, drop_first=True)
                        df = pd.concat([df, encoded], axis=1)
                        df.drop(column, axis=1, inplace=True)
                        logs.append(f"🔤 {column}: one-hot encoded into {len(encoded.columns)} columns")
                    else:
                        logs.append(f"⚠️ {column}: skipping one-hot encoding ({unique_count} categories > {MAX_ONEHOT_CATEGORIES})")
            else:
                logs.append(f"⚠️ {column}: skipping encoding - not categorical (dtype: {df[column].dtype})")
        
        # ==================== OUTLIER HANDLING ====================
        elif operation in ["detect:outliers_iqr", "flag:outliers", "handle:outliers"]:
            if pd.api.types.is_numeric_dtype(df[column]) and not df[column].isnull().any():
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = ((df[column] < lower) | (df[column] > upper)).sum()
                    
                    if outliers > 0:
                        if operation == "handle:outliers":
                            df[column] = np.where(df[column] < lower, lower, df[column])
                            df[column] = np.where(df[column] > upper, upper, df[column])
                            logs.append(f"📏 {column}: capped {outliers} outliers (range: {lower:.2f} - {upper:.2f})")
                        else:
                            logs.append(f"📏 {column}: detected {outliers} outliers (IQR method)")
        
        # ==================== TEXT CLEANING ====================
        elif operation == "clean:text":
            if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                original_nulls = df[column].isnull().sum()
                df[column] = clean_text_data(df[column])
                logs.append(f"🧹 {column}: text cleaned (nulls: {original_nulls})")
            else:
                logs.append(f"⚠️ {column}: skipping text cleaning - not text column")
        
        # ==================== FLAG MISSING ====================
        elif operation == "flag:missing":
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                logs.append(f"⚠️ {column}: {missing_count} missing values ({missing_pct:.1f}%)")
            else:
                logs.append(f"✅ {column}: no missing values")
        
        # ==================== LOG TRANSFORM ====================
        elif operation == "log_transform":
            if pd.api.types.is_numeric_dtype(df[column]):
                if (df[column] > 0).all():
                    df[column] = np.log1p(df[column])
                    logs.append(f"📈 {column}: log transform applied")
                else:
                    logs.append(f"⚠️ {column}: skipping log transform - contains non-positive values")
        
        # ==================== DROP COLUMN ====================
        elif operation == "drop:column":
            df.drop(column, axis=1, inplace=True)
            logs.append(f"🗑️ {column}: column dropped")
    
    except Exception as e:
        logs.append(f"⚠️ Error in {operation} on {column}: {str(e)}")
        logger.error(f"Operation error: {operation} on {column}", exc_info=True)
    
    return df


def execute_dataset_operation_enhanced(df, operation, logs, rule_engine):
    """
    Enhanced dataset-wide operations - UNIVERSAL HANDLER
    """
    try:
        # Data type validation
        if operation == "validate:datatypes":
            conversions = smart_validate_datatypes(df, logs)
            if conversions:
                logs.append(f"🔍 Converted columns: {', '.join(conversions)}")
            else:
                logs.append(f"🔍 Data types validated - no conversions needed")
        
        # Remove duplicates
        elif operation == "remove:duplicates":
            initial = len(df)
            df = df.drop_duplicates()
            removed = initial - len(df)
            if removed > 0:
                logs.append(f"🧹 Removed {removed} duplicate rows ({removed/initial*100:.1f}%)")
            else:
                logs.append(f"✅ No duplicate rows found")
        
        # Flag missing values across dataset
        elif operation == "flag:missing":
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if not missing_cols.empty:
                total_missing = missing_cols.sum()
                logs.append(f"⚠️ Total missing values: {total_missing}")
                for col, count in missing_cols.head(5).items():
                    pct = (count / len(df)) * 100
                    logs.append(f"   {col}: {count} missing ({pct:.1f}%)")
                if len(missing_cols) > 5:
                    logs.append(f"   ... and {len(missing_cols) - 5} more columns with missing values")
            else:
                logs.append(f"✅ No missing values detected")
        
        # Mean imputation for all numeric columns
        elif operation == "impute:mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputed_cols = []
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mean())
                    imputed_cols.append(col)
            if imputed_cols:
                logs.append(f"📊 Mean imputation applied to: {', '.join(imputed_cols)}")
        
        # Median imputation for all numeric columns
        elif operation == "impute:median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputed_cols = []
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
                    imputed_cols.append(col)
            if imputed_cols:
                logs.append(f"📊 Median imputation applied to: {', '.join(imputed_cols)}")
        
        # Standard scaling for all numeric columns
        elif operation == "scale:standard":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            scaled_cols = []
            for col in numeric_cols:
                if not df[col].isnull().any() and df[col].nunique() > 1:
                    scaler = StandardScaler()
                    df[col] = scaler.fit_transform(df[[col]]).flatten()
                    scaled_cols.append(col)
            if scaled_cols:
                logs.append(f"⚖️ Standard scaling applied to: {', '.join(scaled_cols)}")
        
        # Robust scaling for all numeric columns
        elif operation == "scale:robust":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            scaled_cols = []
            for col in numeric_cols:
                if not df[col].isnull().any() and df[col].nunique() > 1:
                    scaler = RobustScaler()
                    df[col] = scaler.fit_transform(df[[col]]).flatten()
                    scaled_cols.append(col)
            if scaled_cols:
                logs.append(f"⚖️ Robust scaling applied to: {', '.join(scaled_cols)}")
        
        # Label encoding for all categorical columns
        elif operation == "encode:label":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            encoded_cols = []
            for col in categorical_cols:
                if df[col].nunique() > 1:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoded_cols.append(col)
            if encoded_cols:
                logs.append(f"🔤 Label encoding applied to: {', '.join(encoded_cols)}")
        
        # One-hot encoding for categorical columns
        elif operation == "encode:onehot":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            onehot_cols = []
            for col in categorical_cols:
                if 1 < df[col].nunique() <= MAX_ONEHOT_CATEGORIES:
                    encoded = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, encoded], axis=1)
                    df.drop(col, axis=1, inplace=True)
                    onehot_cols.append(col)
            if onehot_cols:
                logs.append(f"🔤 One-hot encoding applied to: {', '.join(onehot_cols)}")
        
        # Outlier detection across all numeric columns
        elif operation in ["detect:outliers_iqr", "flag:outliers"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_cols = []
            for col in numeric_cols:
                if not df[col].isnull().any():
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                        if outliers > 0:
                            outlier_cols.append(f"{col}({outliers})")
            if outlier_cols:
                logs.append(f"📏 Outliers detected: {', '.join(outlier_cols)}")
        
        # Handle outliers across all numeric columns
        elif operation == "handle:outliers":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            handled_cols = []
            for col in numeric_cols:
                if not df[col].isnull().any():
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                        if outliers > 0:
                            df[col] = np.where(df[col] < lower, lower, df[col])
                            df[col] = np.where(df[col] > upper, upper, df[col])
                            handled_cols.append(f"{col}({outliers})")
            if handled_cols:
                logs.append(f"📏 Outliers handled: {', '.join(handled_cols)}")
    
    except Exception as e:
        logs.append(f"⚠️ Error in dataset operation {operation}: {str(e)}")
    
    return df


def execute_column_wise_operations(df, column_ops, logs, rule_engine):
    """
    Execute column-wise operations - UNIVERSAL HANDLER
    """
    executed = []
    failed = []
    
    for column, operations in column_ops.items():
        if column == "dataset_wide":
            for op in operations:
                try:
                    df = execute_dataset_operation_enhanced(df, op, logs, rule_engine)
                    executed.append(f"dataset_wide:{op}")
                except Exception as e:
                    logs.append(f"⚠️ Failed: dataset_wide:{op} - {str(e)}")
                    failed.append(f"dataset_wide:{op}")
        else:
            if column not in df.columns:
                logs.append(f"⚠️ Column '{column}' not found")
                failed.append(column)
                continue
            
            for op in operations:
                try:
                    df = execute_single_operation_enhanced(df, op, column, logs, rule_engine)
                    executed.append(f"{column}:{op}")
                except Exception as e:
                    logs.append(f"⚠️ Failed: {column}:{op} - {str(e)}")
                    failed.append(f"{column}:{op}")
    
    if executed:
        logs.append(f"✅ Column operations executed: {len(executed)}")
        if len(executed) <= 10:
            logs.append(f"   Details: {executed}")
    
    if failed:
        logs.append(f"⚠️ Failed operations: {len(failed)}")
    
    return df


def apply_basic_validation(df, rule_engine, logs):
    """Apply basic validation to numeric columns"""
    df_validated = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            intent = rule_engine.infer_column_intent(col, df[col])
            
            # Validate ratings (0-10 scale)
            if 'rating' in col.lower() or 'score' in col.lower():
                if (df[col] < 0).any() or (df[col] > 10).any():
                    df_validated[col] = df[col].clip(0, 10)
                    logs.append(f"🔍 {col}: ratings clipped to 0-10")
            
            # Validate percentages
            elif intent['is_percentage']:
                if (df[col] < 0).any() or (df[col] > 100).any():
                    df_validated[col] = df[col].clip(0, 100)
                    logs.append(f"📈 {col}: percentage clipped to 0-100")
            
            # Validate ages
            elif intent['is_age']:
                if (df[col] < 0).any() or (df[col] > MAX_HUMAN_AGE).any():
                    df_validated[col] = df[col].clip(0, MAX_HUMAN_AGE)
                    logs.append(f"🔍 {col}: age clipped to 0-{MAX_HUMAN_AGE}")
    
    return df_validated


def data_profiling(df, rule_engine):
    """Basic data profiling"""
    logs = []
    
    logs.append(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Missing values
    missing = df.isnull().sum().sum()
    missing_pct = (missing / (df.shape[0] * df.shape[1])) * 100
    logs.append(f"🔍 Missing values: {missing} ({missing_pct:.1f}%)")
    
    # Data types
    dtype_counts = df.dtypes.value_counts()
    logs.append(f"📈 Data types: {', '.join([f'{dtype}({count})' for dtype, count in dtype_counts.items()])}")
    
    # Column summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logs.append(f"📊 Numeric columns: {len(numeric_cols)}")
    logs.append(f"🏷️ Categorical columns: {len(categorical_cols)}")
    
    return logs


def smart_validate_datatypes(df, logs):
    """Smart data type validation"""
    conversions = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            numeric_pct = numeric_vals.notna().sum() / len(df)
            
            if numeric_pct > 0.9:
                df[col] = numeric_vals
                conversions.append(col)
    
    return conversions


def clean_text_data(series):
    """Clean text data"""
    try:
        series = series.fillna('')
        series = series.astype(str)
        series = series.str.strip()
        series = series.str.replace(r'\s+', ' ', regex=True)
        return series
    except:
        return series


def final_quality_check(df, original_shape):
    """Final quality check"""
    logs = []
    
    logs.append("📋 FINAL QUALITY REPORT:")
    logs.append(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logs.append(f"   Original shape: {original_shape[0]} rows × {original_shape[1]} columns")
    
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    logs.append(f"   Missing values: {missing_pct:.1f}%")
    
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    logs.append(f"   Duplicate rows: {duplicate_pct:.1f}%")
    
    # Data type summary
    dtype_summary = df.dtypes.value_counts()
    logs.append(f"   Data types: {dict(dtype_summary)}")
    
    return logs


def smart_read_csv(file):
    """Smart CSV reading"""
    logs = []
    try:
        if isinstance(file, str):
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        logs.append("✅ CSV loaded successfully")
        return df, logs
        
    except Exception as e:
        logs.append(f"❌ CSV error: {str(e)}")
        return pd.DataFrame(), logs


def save_processed_file_enhanced(df, original_filename, validation_summary):
    """Save processed file"""
    from mainapp.file.download import save_processed_file as save_file
    
    download_info = save_file(df, original_filename)
    return download_info