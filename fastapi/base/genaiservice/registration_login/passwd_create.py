from pydantic import BaseModel, validator, field_validator


class User(BaseModel):
    username: str
    password: str

    @field_validator("password")
    def validate(self,password):
        if len(self.password) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(char.isdigit() for char in self.password):
            raise ValueError("password must be at least one digit.")
        if not any(char.isupper() for char in self.password) < 3:
            raise ValueError("password must be at least 3 characters long.")
        return password


@app.post("/create")
def create_user(user: User):
    return {"username": user.username, "Message": "Account successfully created"}
