import requests
import os

def download_file(url, filename, path=''):
    path = os.path.abspath(path)  # Ensure the path is absolute
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raises stored HTTPError, if one occurred.
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            chunk_size = 8192  # 8KB chunks
            downloaded_size = 0

            with open(full_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        print(f"Downloaded {downloaded_size} of {total_size_in_bytes} bytes", end='\r')
            if downloaded_size != total_size_in_bytes:
                print("\nWarning: Downloaded file size does not match expected content length. Download might be incomplete.")
            else:
                print("\nDownload completed successfully.")
    except requests.exceptions.RequestException as e:
        print(f"\nError during download: {e}")
