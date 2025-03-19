from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema

# Schemas für die API-Dokumentation
file_upload_schema = swagger_auto_schema(
    operation_description="Upload a file to be used with AI models",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['file'],
        properties={
            'file': openapi.Schema(
                type=openapi.TYPE_FILE,
                description="File to upload"
            )
        }
    ),
    responses={
        201: "File uploaded successfully",
        400: "Bad request",
        429: "Too many requests"
    }
)

file_detail_schema = swagger_auto_schema(
    operation_description="Get details of a specific file",
    responses={
        200: "File details retrieved successfully",
        404: "File not found"
    }
)

file_delete_schema = swagger_auto_schema(
    operation_description="Delete a specific file",
    responses={
        204: "File deleted successfully",
        404: "File not found"
    }
)

file_process_schema = swagger_auto_schema(
    operation_description="Process a file with AI",
    responses={
        200: "File processed successfully",
        404: "File not found"
    }
) 