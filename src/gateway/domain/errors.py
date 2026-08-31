class GatewayError(Exception):
    code = "gateway_error"
    error_type = "server_error"
    status_code = 500
    retryable = False
    default_message = "The gateway could not complete the request."

    def __init__(
        self,
        message: str | None = None,
        *,
        upstream_status: int | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.upstream_status = upstream_status
        super().__init__(self.message)


class InvalidRequestError(GatewayError):
    code = "invalid_request"
    error_type = "invalid_request_error"
    status_code = 400


class AuthenticationError(GatewayError):
    code = "invalid_api_key"
    error_type = "authentication_error"
    status_code = 401
    default_message = "The supplied gateway API key is invalid."


class ModelNotFoundError(GatewayError):
    code = "model_not_found"
    error_type = "invalid_request_error"
    status_code = 404


class UpstreamAuthenticationError(GatewayError):
    code = "upstream_authentication_failed"
    status_code = 502
    default_message = "The upstream provider rejected the gateway credentials."


class UpstreamRateLimitedError(GatewayError):
    code = "upstream_rate_limited"
    error_type = "rate_limit_error"
    status_code = 429
    retryable = True
    default_message = "The upstream provider is rate limited."


class UpstreamUnavailableError(GatewayError):
    code = "upstream_unavailable"
    status_code = 503
    retryable = True
    default_message = "The upstream provider is unavailable."


class UpstreamConnectTimeoutError(GatewayError):
    code = "upstream_connect_timeout"
    status_code = 504
    retryable = True
    default_message = "Timed out while connecting to the upstream provider."


class UpstreamFirstTokenTimeoutError(GatewayError):
    code = "upstream_first_token_timeout"
    status_code = 504
    retryable = True
    default_message = "The upstream provider did not begin responding in time."


class UpstreamIdleTimeoutError(GatewayError):
    code = "upstream_idle_timeout"
    status_code = 504
    retryable = True
    default_message = "The upstream provider stopped producing events."


class UpstreamTotalTimeoutError(GatewayError):
    code = "upstream_total_timeout"
    status_code = 504
    retryable = True
    default_message = "The upstream request exceeded its total time limit."


class UpstreamProtocolError(GatewayError):
    code = "upstream_protocol_error"
    status_code = 502
    default_message = "The upstream provider returned an invalid event stream."
