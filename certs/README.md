# Client Certificates Directory

This directory should contain SSL certificates for secure communication with the face recognition API server.

## Supported Certificate Formats:

### Option 1: Android Client Format (Preferred)
- `deep_face_server_v2.pem` - Server certificate in PEM format (for client authentication)
- `deep_face_server.crt` - Server certificate in CRT format (for SSL verification)

### Option 2: Standard Format (Fallback)
- `client.crt` - Client certificate file
- `client.key` - Client private key file  
- `ca.crt` - Certificate Authority (CA) certificate file

## Certificate Priority:

1. The application first looks for Android client format (`deep_face_server_v2.pem` + `deep_face_server.crt`)
2. If not found, falls back to standard format (`client.crt` + `client.key` + `ca.crt`)
3. If certificates are found but SSL verification fails, authentication is attempted with SSL verification disabled

## Security Notice:

- The application will work without certificates but will use insecure HTTP connections
- For production use, obtain proper certificates from your API provider
- Never commit certificate files to version control
- Ensure certificate files have appropriate permissions (readable by application only)

## Certificate Setup:

1. Obtain certificates from your face recognition API provider
2. Place the certificate files in this directory using either format above
3. Restart the application to use secure connections

## Android Client Compatibility:

If your Android clients are working with `deep_face_server_v2.pem` and `deep_face_server.crt`, place those same files in this directory.

### Usage Options:

- **Both files**: Maximum compatibility with client authentication and SSL verification
- **PEM only**: Client authentication without SSL verification  
- **CRT only**: SSL verification without client authentication

The application will automatically detect which files are available and configure SSL accordingly.