def fib(n):
  a, b = 0, 1
  while a < n:
    print(a, end = " ")
    a, b = b, a + b
  print()


def ask_ok(prompt, retries=4, reminder='Please try again!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)


def test1():
    ask_ok('Do you really want to quit?')
    ask_ok('OK to overwrite the file?', 2)
    ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')


def info(name, no = 10):
    print(f"{name}, {no}")

def test2():
    info(name = "yaojingguo", no=100)
    print()
    info(no=100, name = "yaojingguo")

def function(a):
    pass


#  function(0, a = 0)
def cheeseshop(kind, *arguments, **keywords):
    print("-- Do you have any", kind, "?")
    print("-- I'm sorry, we're all out of", kind)
    for arg in arguments:
        print(arg)
    print("-" * 40)
    for kw in keywords:
        print(kw, ":", keywords[kw])

#  cheeseshop("Limburger", "It's very runny, sir.",
#             "It's really very, VERY runny, sir.",
#             shopkeeper="Michael Palin",
#             client="John Cleese",
#             sketch="Cheese Shop Sketch")

def standard_arg(arg):
    print(arg)


def pos_only_arg(arg, /):
    print(arg)


def kwd_only_arg(*, arg):
    print(arg)

def combined_example(pos_only, /, standard, *, kwd_only):
    print(pos_only, standard, kwd_only)

def concat(*args, sep="/"):
    return sep.join(args)


def foo(name, **kwds):
    print(kwds)
    return 'name' in kwds
