import httpx
import ssl
import os
http_client = None

def main():
    # Create a custom SSL context for ZScaler
    ssl_context = ssl.create_default_context()

    # Load the combined certificate bundle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cert_bundle_path = os.path.join(current_dir, "combined_cacert.pem")
    ssl_context.load_verify_locations(cert_bundle_path)

    # Configure SSL context for corporate proxy (ZScaler compatibility)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Create httpx client with custom SSL context
    http_client = httpx.Client(verify=ssl_context)
    
    return http_client