from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel

# to get a string like this run:
# openssl rand -hex 32
# 用于签名和验证 JWT 的密钥，必须保密，生产环境应通过环境变量配置
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
# 指定 JWT 使用的签名算法为 HS256（HMAC + SHA-256），这是一种对称加密算法
ALGORITHM = "HS256"

## 访问令牌过期分钟数
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[str] = None


class UserInDB(User):
    hashed_password: str


password_hash = PasswordHash.recommended()

DUMMY_HASH = password_hash.hash("dummypassword")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    """
    验证明文密码与哈希密码是否匹配

    Args:
        plain_password: 用户输入的明文密码
        hashed_password: 数据库中存储的哈希密码

    Returns:
        bool: 密码匹配返回 True，否则返回 False
    """
    return password_hash.verify(plain_password, hashed_password)


def get_password_hash(password):
    """
    对明文密码进行哈希加密

    Args:
        password: 需要加密的明文密码

    Returns:
        str: 加密后的哈希密码字符串
    """
    return password_hash.hash(password)


def get_user(db, username: str):
    """
    从数据库中获取指定用户名的用户信息

    Args:
        db: 用户数据库字典
        username: 要查询的用户名

    Returns:
        UserInDB: 如果用户存在，返回 UserInDB 对象；否则返回 None
    """
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    """
    验证用户凭据（用户名和密码）

    Args:
        fake_db: 用户数据库字典
        username: 用户名
        password: 明文密码

    Returns:
        UserInDB or bool: 验证成功返回 UserInDB 对象，失败返回 False
    """
    user = get_user(fake_db, username)
    if not user:
        verify_password(password, DUMMY_HASH)
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta:  Optional[timedelta] = None):
    """
    创建 JWT access token

    Args:
        data: 要编码到 token 中的数据字典，通常包含用户标识
        expires_delta: token 过期时间的增量，如果为 None 则默认 15 分钟

    Returns:
        str: 编码后的 JWT token 字符串
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    从 JWT token 中解析并获取当前用户信息

    Args:
        token: OAuth2 Bearer token 字符串，通过依赖注入自动获取

    Returns:
        UserInDB: 当前认证用户的 UserInDB 对象

    Raises:
        HTTPException: 当 token 无效、过期或用户不存在时抛出 401 错误
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 这里解析token时，自动检查是否过期
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
        current_user: Annotated[User, Depends(get_current_user)],
):
    """
    检查当前用户是否为活跃状态

    Args:
        current_user: 已通过 token 验证的用户对象，通过依赖注入自动获取

    Returns:
        User: 如果用户活跃，返回用户对象

    Raises:
        HTTPException: 当用户被禁用时抛出 400 错误
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def login_for_access_token(
        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """
    用户登录接口，验证凭据并返回 JWT access token

    Args:
        form_data: OAuth2 密码请求表单，包含 username 和 password 字段

    Returns:
        Token: 包含 access_token 和 token_type 的 Token 对象

    Raises:
        HTTPException: 当用户名或密码错误时抛出 401 错误
    """
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me/")
async def read_users_me(
        current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """
    获取当前登录用户的详细信息

    Args:
        current_user: 当前活跃的认证用户对象，通过依赖注入自动获取

    Returns:
        User: 当前用户的 User 对象
    """
    return current_user


@app.get("/users/me/items/")
async def read_own_items(
        current_user: Annotated[User, Depends(get_current_active_user)],
):
    """
    获取当前登录用户的物品列表

    Args:
        current_user: 当前活跃的认证用户对象，通过依赖注入自动获取

    Returns:
        list: 包含物品信息的列表，每个物品包含 item_id 和 owner 字段
    """
    return [{"item_id": "Foo", "owner": current_user.username}]
