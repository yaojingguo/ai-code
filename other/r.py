import requests


r = requests.get('https://httpbin.org/basic-auth/user/pass', auth=('user', 'pass'))

print(f"{r}")

print(f"{r.status_code}")
print(f"{r.encoding}")
print(f"{r.text}")
print(f"{r.json()}")

print(r.headers["content-type"])
