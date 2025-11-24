class RealTimeDetectionSystem:
    """Real-time evil twin detection system"""
    
    def __init__(self, ensemble_model, preprocessor, interface='wlan0'):
        self.ensemble_model = ensemble_model
        self.preprocessor = preprocessor
        self.interface = interface
        self.scanner = WirelessNetworkScanner(interface)
        self.feature_extractor = FeatureExtractor()
        self.alerts = []
        
    def monitor(self, duration=300, check_interval=30):
        """Monitor network in real-time"""
        print(f"\n[*] Starting real-time monitoring for {duration} seconds...")
        print(f"[*] Checking every {check_interval} seconds...\n")
        
        start_time = time.time()
        iteration = 0
        
        while (time.time() - start_time) < duration:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Capture packets
            networks = self.scanner.start_capture(duration=check_interval)
            
            if not networks:
                print("[!] No networks detected")
                continue
            
            # Extract features
            features_df = self.feature_extractor.extract_features(networks)
            
            # Preprocess
            X_new, _ = self.preprocessor.preprocess(features_df, fit=False)
            
            # Predict
            predictions = self.ensemble_model.predict(X_new)
            probabilities = self.ensemble_model.predict_proba(X_new)
            
            # Check for evil twins
            for idx, (bssid, data) in enumerate(networks.items()):
                if predictions[idx] == 1:  # Evil twin detected
                    alert = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'bssid': bssid,
                        'ssid': data['ssid'],
                        'channel': data['channel'],
                        'signal_strength': np.mean(data['signal_strength']),
                        'confidence': probabilities[idx] * 100,
                        'crypto': data['crypto']
                    }
                    
                    self.alerts.append(alert)
                    self.raise_alert(alert)
            
            time.sleep(1)
        
        print(f"\n[+] Monitoring completed. Total alerts: {len(self.alerts)}")
        return self.alerts
    
    def raise_alert(self, alert):
        """Raise alert for detected evil twin"""
        print("\n" + "="*60)
        print("🚨 EVIL TWIN ATTACK DETECTED!")
        print("="*60)
        print(f"Timestamp:       {alert['timestamp']}")
        print(f"BSSID:           {alert['bssid']}")
        print(f"SSID:            {alert['ssid']}")
        print(f"Channel:         {alert['channel']}")
        print(f"Signal Strength: {alert['signal_strength']:.2f} dBm")
        print(f"Confidence:      {alert['confidence']:.2f}%")
        print(f"Encryption:      {alert['crypto']}")
        print("="*60)
        print("⚠️  RECOMMENDED ACTIONS:")
        print("  1. Do not connect to this network")
        print("  2. Alert network administrator")
        print("  3. Monitor for similar attacks")
        print("="*60 + "\n")
    
    def generate_report(self, filename='detection_report.txt'):
        """Generate detection report"""
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVIL TWIN DETECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Alerts: {len(self.alerts)}\n\n")
            
            for idx, alert in enumerate(self.alerts, 1):
                f.write(f"\nAlert #{idx}\n")
                f.write("-"*40 + "\n")
                for key, value in alert.items():
                    f.write(f"{key.capitalize():20s}: {value}\n")
                f.write("-"*40 + "\n")
        
        print(f"[+] Report saved to {filename}")

# Example usage (requires wireless interface)
# detection_system = RealTimeDetectionSystem(ensemble_detector, preprocessor, interface='wlan0')
# alerts = detection_system.monitor(duration=300, check_interval=30)
# detection_system.generate_report()