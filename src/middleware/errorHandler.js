/**
 * Error Handler Middleware - Centralized error handling for API
 */

export function errorHandler(logger) {
  return (error, req, res, next) => {
    // Log the error
    logger.error('Unhandled error', {
      error: error.message,
      stack: error.stack,
      url: req.url,
      method: req.method,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      requestId: req.id
    });

    // Default error response
    let statusCode = 500;
    let errorResponse = {
      error: 'Internal Server Error',
      message: 'An unexpected error occurred',
      timestamp: new Date().toISOString(),
      requestId: req.id
    };

    // Handle specific error types
    if (error.name === 'ValidationError') {
      statusCode = 400;
      errorResponse.error = 'Validation Error';
      errorResponse.message = error.message;
    } else if (error.name === 'UnauthorizedError') {
      statusCode = 401;
      errorResponse.error = 'Unauthorized';
      errorResponse.message = 'Authentication required';
    } else if (error.name === 'ForbiddenError') {
      statusCode = 403;
      errorResponse.error = 'Forbidden';
      errorResponse.message = 'Insufficient permissions';
    } else if (error.name === 'NotFoundError') {
      statusCode = 404;
      errorResponse.error = 'Not Found';
      errorResponse.message = error.message;
    } else if (error.name === 'ConflictError') {
      statusCode = 409;
      errorResponse.error = 'Conflict';
      errorResponse.message = error.message;
    } else if (error.name === 'TooManyRequestsError') {
      statusCode = 429;
      errorResponse.error = 'Too Many Requests';
      errorResponse.message = 'Rate limit exceeded';
    }

    // Handle Joi validation errors
    if (error.isJoi) {
      statusCode = 400;
      errorResponse.error = 'Validation Error';
      errorResponse.message = error.details[0].message;
      errorResponse.details = error.details;
    }

    // Handle syntax errors (malformed JSON)
    if (error instanceof SyntaxError && error.status === 400 && 'body' in error) {
      statusCode = 400;
      errorResponse.error = 'Bad Request';
      errorResponse.message = 'Invalid JSON in request body';
    }

    // In development, include stack trace
    if (process.env.NODE_ENV === 'development') {
      errorResponse.stack = error.stack;
    }

    res.status(statusCode).json(errorResponse);
  };
}