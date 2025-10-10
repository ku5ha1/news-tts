#!/usr/bin/env python3
"""
Script to fix IndicTrans2 compatibility issues
This script helps diagnose and fix the "not enough values to unpack" error
"""

import sys
import subprocess
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_indictrans_version():
    """Check which version of IndicTransToolkit is installed"""
    try:
        import IndicTransToolkit
        logger.info(f"IndicTransToolkit version: {IndicTransToolkit.__version__}")
        return True
    except ImportError:
        logger.error("IndicTransToolkit not found")
        return False
    except AttributeError:
        logger.info("IndicTransToolkit found but no version info")
        return True

def check_indictrans2():
    """Check if IndicTrans2 is available"""
    try:
        import indictrans2
        logger.info(f"IndicTrans2 version: {indictrans2.__version__}")
        return True
    except ImportError:
        logger.error("IndicTrans2 not found")
        return False
    except AttributeError:
        logger.info("IndicTrans2 found but no version info")
        return True

def test_preprocessing():
    """Test the preprocessing functionality"""
    try:
        from IndicTransToolkit import IndicProcessor
        
        # Initialize processor
        ip = IndicProcessor(inference=True)
        
        # Test preprocessing
        test_text = "Hello world"
        batch = ip.preprocess_batch([test_text], src_lang="eng_Latn", tgt_lang="hin_Deva")
        
        logger.info(f"Preprocessing test successful. Return type: {type(batch)}")
        logger.info(f"Return value: {batch}")
        
        # Test postprocessing
        decoded = ["नमस्ते दुनिया"]
        translations = ip.postprocess_batch(decoded, lang="hin_Deva")
        
        logger.info(f"Postprocessing test successful. Return type: {type(translations)}")
        logger.info(f"Return value: {translations}")
        
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        return False

def install_indictrans_toolkit():
    """Install IndicTransToolkit"""
    try:
        logger.info("Installing IndicTransToolkit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "IndicTransToolkit>=1.0.0"])
        logger.info("IndicTransToolkit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install IndicTransToolkit: {e}")
        return False

def main():
    """Main function to diagnose and fix issues"""
    logger.info("=== IndicTrans2 Compatibility Check ===")
    
    # Check current installations
    logger.info("1. Checking current installations...")
    has_toolkit = check_indictrans_version()
    has_indictrans2 = check_indictrans2()
    
    # Test preprocessing
    logger.info("2. Testing preprocessing functionality...")
    preprocessing_works = test_preprocessing()
    
    if not preprocessing_works:
        logger.warning("Preprocessing test failed. This is likely the source of your error.")
        
        if not has_indictrans2:
            logger.info("3. Installing IndicTransToolkit...")
            if install_indictrans_toolkit():
                logger.info("4. Re-testing preprocessing...")
                preprocessing_works = test_preprocessing()
                
                if preprocessing_works:
                    logger.info("✅ SUCCESS: IndicTransToolkit installation fixed the issue!")
                else:
                    logger.error("❌ FAILED: Issue persists after IndicTransToolkit installation")
            else:
                logger.error("❌ FAILED: Could not install IndicTransToolkit")
        else:
            logger.error("❌ FAILED: IndicTransToolkit is installed but preprocessing still fails")
    else:
        logger.info("✅ SUCCESS: Preprocessing is working correctly!")
    
    logger.info("=== Check Complete ===")

if __name__ == "__main__":
    main()
