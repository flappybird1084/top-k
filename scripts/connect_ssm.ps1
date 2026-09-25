<#
.SYNOPSIS
Check and open an AWS Systems Manager Session Manager shell to the EC2 host.
.EXAMPLE
./scripts/connect_ssm.ps1 -InstanceId i-0123456789abcdef0 -Region us-west-2 -Profile top-k
.EXAMPLE
./scripts/connect_ssm.ps1 -InstanceId i-0123456789abcdef0 -Region us-west-2 -CheckOnly
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^i-[0-9a-f]{8}([0-9a-f]{9})?$')]
    [string]$InstanceId,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z]{2}(-gov)?-[a-z]+-\d+$')]
    [string]$Region,
    [string]$Profile,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw 'AWS CLI is missing. Install AWS CLI v2 from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html'
}
$awsArgs = @('--region', $Region, '--no-cli-pager')
if ($Profile) { $awsArgs += @('--profile', $Profile) }

try {
    $identity = (& aws sts get-caller-identity @awsArgs --output json | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $identity.Account) { throw 'AWS identity lookup failed' }
} catch {
    throw "AWS authentication failed. Log in to the selected profile with 'aws sso login --profile <name>' or configure credentials. $($_.Exception.Message)"
}
Write-Host "AWS account: $($identity.Account); principal: $($identity.Arn)"

try {
    $instance = (& aws ec2 describe-instances --instance-ids $InstanceId @awsArgs --output json | ConvertFrom-Json).Reservations.Instances | Select-Object -First 1
    if ($LASTEXITCODE -ne 0 -or -not $instance) { throw 'Instance lookup failed' }
} catch {
    throw "EC2 instance $InstanceId was not found in $Region or access was denied. $($_.Exception.Message)"
}
Write-Host "EC2 state: $($instance.State.Name); instance profile: $($instance.IamInstanceProfile.Arn)"
if ($instance.State.Name -ne 'running') {
    throw "EC2 instance $InstanceId is not running"
}
if (-not $instance.IamInstanceProfile.Arn) {
    throw 'EC2 has no instance profile. Attach an IAM role with AmazonSSMManagedInstanceCore.'
}

try {
    $connection = (& aws ssm get-connection-status --target $InstanceId @awsArgs --output json | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $connection.Status) { throw 'Connection lookup failed' }
} catch {
    throw "Could not check SSM connection status. Check IAM permissions for ssm:GetConnectionStatus. $($_.Exception.Message)"
}
Write-Host "SSM connection: $($connection.Status)"
if ($connection.Status -ne 'connected') {
    throw 'SSM Agent is not connected. Check the instance role, agent service, and outbound access to SSM endpoints.'
}
if ($CheckOnly) { return }
if (-not (Get-Command session-manager-plugin -ErrorAction SilentlyContinue)) {
    throw 'Session Manager plugin is missing. Install it from https://docs.aws.amazon.com/systems-manager/latest/userguide/install-plugin-windows.html'
}

Write-Host "Opening Session Manager shell on $InstanceId..."
& aws ssm start-session --target $InstanceId @awsArgs
if ($LASTEXITCODE -ne 0) { throw "SSM session failed with exit code $LASTEXITCODE" }
