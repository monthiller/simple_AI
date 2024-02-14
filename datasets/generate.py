def light_on_after_12():
  dataset = {
    "inputs":[],
    "outputs":[],
    "tolerance":[],
  }
  for i in range(50):
    input = 24*i/50
    if input<=12:
      dataset["inputs"].append([input,])
      dataset["outputs"].append([0,])
      dataset["tolerance"].append([0, 0.4])
      continue
    dataset["inputs"].append([input,])
    dataset["outputs"].append([1])
    dataset["tolerance"].append([0.6, 1])
  return dataset

def light_on_after_19():
  dataset = {
    "inputs":[],
    "outputs":[],
    "tolerance":[],
  }
  for i in range(50):
    input = 24*i/50
    if input<=19:
      dataset["inputs"].append([input,])
      dataset["outputs"].append([0,])
      dataset["tolerance"].append([0, 0.4])
      continue
    dataset["inputs"].append([input,])
    dataset["outputs"].append([1])
    dataset["tolerance"].append([0.6, 1])
  return dataset
