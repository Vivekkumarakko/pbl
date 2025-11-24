def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic dataset for testing"""
    np.random.seed(42)
    
    # Legitimate networks
    n_legitimate = int(n_samples * 0.7)
    legitimate_data = {
        'avg_signal_strength': np.random.normal(-60, 10, n_legitimate),
        'std_signal_strength': np.random.uniform(2, 8, n_legitimate),
        'min_signal_strength': np.random.normal(-75, 5, n_legitimate),
        'max_signal_strength': np.random.normal(-50, 5, n_legitimate),
        'signal_variance': np.random.uniform(4, 50, n_legitimate),
        'channel': np.random.choice([1, 6, 11], n_legitimate),
        'beacon_count': np.random.randint(50, 200, n_legitimate),
        'beacon_rate': np.random.uniform(8, 12, n_legitimate),
        'mac_vendor': np.random.choice([1, 2, 3], n_legitimate),
        'crypto_type': np.random.choice([1, 2], n_legitimate),  # WPA2, WPA3
        'ssid_length': np.random.randint(5, 20, n_legitimate),
        'ssid_has_spaces': np.random.choice([0, 1], n_legitimate),
        'ssid_special_chars': np.random.randint(0, 2, n_legitimate),
        'uptime': np.random.uniform(3600, 86400, n_legitimate),
        'signal_strength_range': np.random.uniform(10, 25, n_legitimate),
        'label': 0  # Legitimate
    }
    
    # Evil twin networks (malicious)
    n_evil = n_samples - n_legitimate
    evil_data = {
        'avg_signal_strength': np.random.normal(-55, 5, n_evil),  # Typically stronger
        'std_signal_strength': np.random.uniform(8, 15, n_evil),  # More variance
        'min_signal_strength': np.random.normal(-70, 8, n_evil),
        'max_signal_strength': np.random.normal(-45, 3, n_evil),
        'signal_variance': np.random.uniform(60, 150, n_evil),  # Higher variance
        'channel': np.random.choice([1, 6, 11], n_evil),
        'beacon_count': np.random.randint(20, 100, n_evil),  # Lower beacon count
        'beacon_rate': np.random.uniform(5, 9, n_evil),  # Lower rate
        'mac_vendor': np.random.choice([0, 4], n_evil),  # Unknown/suspicious vendors
        'crypto_type': np.random.choice([0, 1], n_evil),  # Often open or weak encryption
        'ssid_length': np.random.randint(3, 25, n_evil),
        'ssid_has_spaces': np.random.choice([0, 1], n_evil),
        'ssid_special_chars': np.random.randint(2, 8, n_evil),  # More special chars
        'uptime': np.random.uniform(60, 7200, n_evil),  # Shorter uptime
        'signal_strength_range': np.random.uniform(20, 40, n_evil),  # Higher range
        'label': 1  # Evil Twin
    }
    
    # Combine datasets
    df_legitimate = pd.DataFrame(legitimate_data)
    df_evil = pd.DataFrame(evil_data)
    df = pd.concat([df_legitimate, df_evil], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# Generate dataset
dataset = generate_synthetic_dataset(n_samples=2000)
print(f"Dataset shape: {dataset.shape}")
print(f"\nClass distribution:\n{dataset['label'].value_counts()}")
print(f"\nDataset preview:\n{dataset.head()}")