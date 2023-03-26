def normalize(target):
    arr = []
    for i in range(len(target)):
        arr.append(target[i])

    normalized = []
    for i in range(len(arr)):
        normalized.append((arr[i] - min(arr)) / (max(arr) - min(arr)))
    return normalized

def normalize1(target):
    arr = []
    for i in range(len(target)):
        arr.append(target[i][0])

    normalized = []
    for i in range(len(arr)):
        normalized.append([(arr[i] - min(arr)) / (max(arr) - min(arr))])
    return normalized

def denormalize(normalized, target):
    arr = []
    norm = []
    for i in range(len(target)):
        arr.append(target[i])
    for i in range(len(normalized)):
        norm.append(normalized[i])

    denormalize = []
    for i in range(len(norm)):
        denormalize.append([norm[i][0] * (max(arr)[0] - min(arr)[0]) + min(arr)[0]])
    return denormalize