list1 = [1, 2]
list2 = [10, 20, 30]

for a, b in zip(list1, list2):
  print(a)
  print(b)
  print()

username = "yaojingguo"
print(f"my name: {username}")

print()

no = 100
msg = f"""name: {username}
no: {no}"""
print(msg)

print()

print("name: {}".format(username))
print("name: %s" % (username))
