from typing import Literal

from pydantic import BaseModel, ValidationError

# describe shape using python type hints
class Request(BaseModel):
    text: str
    label: Literal["chat","fact", "reasoning","action"]
    borderline: bool = False

class Batch(BaseModel):
    requests: list[Request]

#build object from python values
r= Request(text="remind me at 5pm", label="action")
print(r)
print(r.label,r.borderline)

#bad data -> ValidationError
# try:
#     Request(text="tell me a joke", label="joke")
# except ValidationError as e:
#     print("Error:", e.errors()[0]['msg'])

# validate JSON text (usually what an api sends back)
raw = '{"requests":[{"text":"hi","label":"chat"},{"text":"anc","label":"fact","borderline":true}]}'

b = Batch.model_validate_json(raw)

print(len(b.requests),b.requests[1].borderline)

#get model's schema either dict or json from a object (instance of that class)
print(r.model_dump())
print(r.model_dump_json())

#schema that gets handed to the model: just allowed labels.
print(Request.model_json_schema()["properties"]["label"]["enum"])