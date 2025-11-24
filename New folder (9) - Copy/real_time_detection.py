"""
Real-Time Evil Twin Detection System
This script monitors real wireless networks and detects evil twin attacks
Requires: Wireless interface in monitor mode and trained models
"""

import sys
import os

# Check if running as root/admin (required for packet capture)
if os.name == 'posix' and os.geteuid() != 0:
    print("[!] ERROR: This script requires root privileges for packet capture")
    print("[!] Please run with: sudo python real_time_detection.py")
    sys.exit(1)

from evil_twin_modules import (
    ModelManager,
    RealTimeDetectionSystem,
    HybridEnsembleDetector,
    DeepLearningDetector
)

def load_models():
    """Load pre-trained models"""
    print("[*] Loading pre-trained models...")
    try:
        rf_model, svm_model, dl_model_wrapper, preprocessor = ModelManager.load_models()
        
        # Create DeepLearningDetector wrapper
        dl_detector = DeepLearningDetector(input_dim=None)
        dl_detector.model = dl_model_wrapper.model
        
        # Create ensemble
        ensemble_detector = HybridEnsembleDetector(rf_model, svm_model, dl_detector)
        
        print("[+] Models loaded successfully!")
        return ensemble_detector, preprocessor
    except FileNotFoundError as e:
        print(f"[!] ERROR: Models not found. Please train models first:")
        print(f"    python evil_twin_detection.py")
        sys.exit(1)
    except Exception as e:
        print(f"[!] ERROR loading models: {e}")
        sys.exit(1)


def main():
    """Main execution for real-time detection"""
    print("="*60)
    print("REAL-TIME EVIL TWIN DETECTION SYSTEM")
    print("="*60)
    
    # Check for wireless interface
    import argparse
    parser = argparse.ArgumentParser(description='Real-time Evil Twin Detection')
    parser.add_argument('-i', '--interface', type=str, default='wlan0',
                       help='Wireless interface name (default: wlan0)')
    parser.add_argument('-d', '--duration', type=int, default=300,
                       help='Monitoring duration in seconds (default: 300)')
    parser.add_argument('-c', '--check-interval', type=int, default=30,
                       help='Check interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Load models
    ensemble_detector, preprocessor = load_models()
    
    # Create detection system
    print(f"\n[*] Initializing detection system on interface: {args.interface}")
    detection_system = RealTimeDetectionSystem(
        ensemble_model=ensemble_detector,
        preprocessor=preprocessor,
        interface=args.interface
    )
    
    # Start monitoring
    try:
        alerts = detection_system.monitor(
            duration=args.duration,
            check_interval=args.check_interval
        )
        
        # Generate report
        if alerts:
            detection_system.generate_report()
            print(f"\n[+] Detection complete. {len(alerts)} alert(s) generated.")
        else:
            print("\n[+] Detection complete. No evil twin attacks detected.")
            
    except KeyboardInterrupt:
        print("\n\n[!] Monitoring interrupted by user")
        if detection_system.alerts:
            detection_system.generate_report()
            print(f"[+] Partial report saved. {len(detection_system.alerts)} alert(s) detected.")
    except Exception as e:
        print(f"\n[!] ERROR during monitoring: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

