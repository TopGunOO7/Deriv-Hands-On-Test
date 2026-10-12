# API rate limits

The public REST API allows 120 requests per minute per API key. A request over the limit receives HTTP 429 and a Retry-After header. The limit is shared across endpoints for the same key. Customers needing higher limits should submit a capacity request.
