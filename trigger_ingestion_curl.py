import requests


def trigger():
    base_url = "http://localhost:8000"
    submit_url = f"{base_url}/rag-tools/crawler-submit/"

    session = requests.Session()

    # 1. Get CSRF Cookie via Admin Login (reliable source)
    print("GET /admin/login/ to fetch cookies...")
    try:
        r = session.get(f"{base_url}/admin/login/")
        print(f"GET Status: {r.status_code}")
    except Exception as e:
        print(f"GET failed: {e}")
        return

    csrf_token = session.cookies.get("csrftoken")
    if not csrf_token:
        print("Error: No CSRF token found in cookies!")
        return
    print(f"CSRF Token: {csrf_token}")

    data = {
        "workflow_id": "ad-hoc",
        "origin_url": "https://de.wikipedia.org/wiki/Bamberg",
        "collection_id": "93758ef2-b0e2-4545-9383-751722026369",
        "mode": "live",
        "review": "required",
        "fetch": "on",
    }
    headers = {
        "X-Tenant-ID": "dev",
        "HX-Request": "true",
        "X-CSRFToken": csrf_token,
        "Referer": f"{base_url}/rag-tools/",
    }

    print(f"POST {submit_url}")
    try:
        response = session.post(submit_url, data=data, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Content: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    trigger()
