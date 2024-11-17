import requests
from concurrent.futures import ThreadPoolExecutor

# Function to send a single request
def send_request(file_path):
    url = "https://craskinsense-be-959292480377.asia-northeast3.run.app/predict"
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://craskinsense.web.app",
        "priority": "u=1, i",
        "referer": "https://craskinsense.web.app/",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
    }
    files = {
        "file": ("IMG_16.png", open(file_path, "rb"), "image/png"),
    }
    try:
        response = requests.post(url, headers=headers, files=files)
        print(f"Response status: {response.status_code}, Response body: {response.text[:20]}")
    except Exception as e:
        print(f"Request failed: {e}")

# Main function to execute concurrent requests
def main(concurrent_requests, file_path):
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request, file_path) for _ in range(concurrent_requests)]
        for future in futures:
            try:
                future.result()  # Wait for each request to complete
            except Exception as e:
                print(f"Error in thread execution: {e}")

# Parameters
CONCURRENT_REQUESTS = 15  # Number of concurrent requests
FILE_PATH = "./Dataset/XTrain/IMG_2.png"  # Path to the file

# Run the script
if __name__ == "__main__":
    main(CONCURRENT_REQUESTS, FILE_PATH)

