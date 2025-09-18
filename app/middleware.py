# app/middleware.py

from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status
from fastapi.responses import JSONResponse

from .database import SessionLocal
from .auth import verify_token, get_user_by_email
from .logger_config import get_logger

logger = get_logger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """
        这个中间件会在每个请求被处理前运行。
        它负责从 token 中解析用户，并将其附加到 request.state。
        """
        request.state.current_user = None

        auth_header = request.headers.get("Authorization")
        token = request.cookies.get("access_token")

        auth_token = None
        if auth_header and auth_header.startswith("Bearer "):
            auth_token = auth_header.split(" ")[1]
        elif token:
            auth_token = token

        if auth_token:
            email = verify_token(auth_token)
            if email:
                db = SessionLocal()
                try:
                    user = get_user_by_email(db, email)
                    if user:
                        request.state.current_user = user
                        logger.debug(f"中间件认证成功: {user.email}")
                finally:
                    db.close()

        response = await call_next(request)
        return response
