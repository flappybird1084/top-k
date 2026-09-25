# EC2 Session Manager access

The production gateway is currently deployed over SSH by
`.github/workflows/production.yml`. Session Manager here provides an operator
shell to the EC2 host; it does not change the deployment workflow.

The EC2 instance needs a running SSM Agent, an attached IAM instance profile
with `AmazonSSMManagedInstanceCore`, and outbound network access to the
regional Systems Manager endpoints. The operator needs IAM permission to
start and terminate sessions on that instance. AWS documents these
prerequisites at:

- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-with-systems-manager-session-manager.html
- https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started.html

On Windows, install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
and the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/install-plugin-windows.html)
from AWS. Configure an AWS profile or sign in with AWS SSO. Then:

```powershell
./scripts/connect_ssm.ps1 -InstanceId i-0123456789abcdef0 -Region us-west-2 -Profile top-k -CheckOnly
./scripts/connect_ssm.ps1 -InstanceId i-0123456789abcdef0 -Region us-west-2 -Profile top-k
```

Omit `-Profile` to use the default credentials. The script verifies caller
identity, EC2 state and instance profile, and live SSM connection status
before opening a shell. It reports the failing prerequisite rather than
trying to connect to an unavailable target. AWS documents the
[`GetConnectionStatus`](https://docs.aws.amazon.com/cli/latest/reference/ssm/get-connection-status.html)
check and [Session Manager troubleshooting](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-troubleshooting.html).
