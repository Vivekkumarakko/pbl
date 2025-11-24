class WirelessNetworkScanner:
    """Capture and analyze 802.11 wireless packets"""
    
    def __init__(self, interface='wlan0'):
        self.interface = interface
        self.packets = []
        self.networks = {}
        
    def packet_handler(self, pkt):
        """Process each captured packet"""
        if pkt.haslayer(Dot11Beacon):
            # Extract network information
            bssid = pkt[Dot11].addr2
            ssid = pkt[Dot11Elt].info.decode('utf-8', errors='ignore')
            
            try:
                # Extract signal strength
                signal_strength = pkt.dBm_AntSignal
            except:
                signal_strength = -100
                
            # Extract channel
            channel = int(ord(pkt[Dot11Elt:3].info))
            
            # Extract encryption info
            crypto = self.get_crypto_type(pkt)
            
            # Store network info
            if bssid not in self.networks:
                self.networks[bssid] = {
                    'ssid': ssid,
                    'signal_strength': [signal_strength],
                    'channel': channel,
                    'crypto': crypto,
                    'first_seen': time.time(),
                    'beacons': 1
                }
            else:
                self.networks[bssid]['signal_strength'].append(signal_strength)
                self.networks[bssid]['beacons'] += 1
                
    def get_crypto_type(self, pkt):
        """Determine encryption type"""
        cap = pkt.sprintf("{Dot11Beacon:%Dot11Beacon.cap%}")
        
        if 'privacy' in cap:
            if pkt.haslayer(Dot11Elt):
                for layer in pkt[Dot11Elt]:
                    if layer.ID == 48:  # RSN Information
                        return 'WPA2'
                return 'WEP'
        return 'Open'
    
    def start_capture(self, duration=60):
        """Capture packets for specified duration"""
        print(f"[*] Starting packet capture on {self.interface} for {duration} seconds...")
        sniff(iface=self.interface, prn=self.packet_handler, timeout=duration)
        print(f"[+] Captured data from {len(self.networks)} networks")
        return self.networks