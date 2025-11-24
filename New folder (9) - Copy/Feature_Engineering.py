class FeatureExtractor:
    """Extract features for ML models"""
    
    def __init__(self):
        self.features = []
        
    def extract_features(self, networks_data):
        """Extract relevant features from network data"""
        feature_list = []
        
        for bssid, data in networks_data.items():
            features = {
                # Signal strength statistics
                'avg_signal_strength': np.mean(data['signal_strength']),
                'std_signal_strength': np.std(data['signal_strength']),
                'min_signal_strength': np.min(data['signal_strength']),
                'max_signal_strength': np.max(data['signal_strength']),
                'signal_variance': np.var(data['signal_strength']),
                
                # Channel information
                'channel': data['channel'],
                
                # Beacon statistics
                'beacon_count': data['beacons'],
                'beacon_rate': data['beacons'] / (time.time() - data['first_seen'] + 1),
                
                # MAC address analysis
                'mac_vendor': self.get_mac_vendor(bssid),
                
                # Encryption
                'crypto_type': data['crypto'],
                
                # SSID characteristics
                'ssid_length': len(data['ssid']),
                'ssid_has_spaces': int(' ' in data['ssid']),
                'ssid_special_chars': sum(not c.isalnum() and c != ' ' for c in data['ssid']),
                
                # Timing analysis
                'uptime': time.time() - data['first_seen'],
                
                # Additional features
                'signal_strength_range': np.max(data['signal_strength']) - np.min(data['signal_strength']),
            }
            
            feature_list.append(features)
            
        return pd.DataFrame(feature_list)
    
    def get_mac_vendor(self, mac):
        """Get MAC vendor OUI (simplified)"""
        # In real implementation, use OUI database
        oui = mac[:8].upper()
        vendor_mapping = {
            '00:0C:41': 1,  # Example: Cisco
            '00:50:56': 2,  # VMware
            # Add more vendors
        }
        return vendor_mapping.get(oui, 0)
