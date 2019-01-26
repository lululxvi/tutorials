raw = (
    "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp. "
    "bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle. "
    "sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj.")
res = ''
for c in raw:
    if c < 'a' or c > 'z':
        res += c
    elif c == 'y':
        res += 'a'
    elif c == 'z':
        res += 'b'
    else:
        res += chr(ord(c) + 2)
print(res)

res = [c if c < 'a' or c > 'z' else 'a' if c == 'y' else 'b' if c == 'z'
       else chr(ord(c) + 2) for c in raw]
res = ''.join(res)
print(res)

def convert(c):
    if c < 'a' or c > 'z':
        return c
    if c == 'y':
        return 'a'
    if c == 'z':
        return 'b'
    return chr(ord(c) + 2)
res = ''.join(map(convert, raw))
print(res)

fr = "abcdefghijklmnopqrstuvwxyz,. '()"
to = "cdefghijklmnopqrstuvwxyzab,. '()"
try:
    table = str.maketrans(fr, to)  # python3
    res = raw.translate(table)
    print(res)
except:
    pass

table = dict(zip(fr, to))
res = ''.join([table[c] for c in raw])
print(res)
