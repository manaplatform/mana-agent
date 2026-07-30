"""AWS EC2 adapter registration point; HTTP transport is injected by configuration."""

from .custom import CustomHTTPProvider


class AWSEC2Provider(CustomHTTPProvider):
    name = "aws"
