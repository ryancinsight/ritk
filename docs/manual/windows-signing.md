# Signing an AsterSlice Windows release

Run **Actions → AsterSlice Métis Package → Run workflow** on the default branch.
The workflow selects the Métis revision from RITK’s lockfile. Its
**AsterSlice-Windows-x64-signed** artifact contains the portable
`app/asterslice.exe`, the per-user `org.ritk.snap.msi` and their SHA-256 inventory.
Signing is configured once by the publisher. Recipients can use the regular
Windows trust settings.

## Configure or change the publisher

1. Create or select a Microsoft Artifact Signing **Public Trust** certificate
   profile for the verified individual or organization that publishes AsterSlice.
   Record the complete certificate subject exactly as Windows displays it.
2. Create the GitHub environment `windows-signing` and restrict deployments to
   the repository’s default branch. Configure an Entra workload identity with the
   GitHub issuer `https://token.actions.githubusercontent.com`, audience
   `api://AzureADTokenExchange` and subject
   `repo:ryancinsight/ritk:environment:windows-signing` (substitute the repository
   owner and name for a fork). Grant that identity the **Artifact Signing
   Certificate Profile Signer** role on the intended certificate profile.
3. Set the following environment variables in GitHub environment settings.
   When changing publishers, select the new verified profile, update its
   matching subject and role assignment, then build a new release.

| GitHub environment variable | Value |
| --- | --- |
| `AZURE_CLIENT_ID` | Entra application or managed identity client ID |
| `AZURE_TENANT_ID` | Entra tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription containing the signing account |
| `WINDOWS_SIGNING_ENDPOINT` | The account’s regional signing endpoint |
| `WINDOWS_SIGNING_ACCOUNT` | Artifact Signing account name |
| `WINDOWS_SIGNING_PROFILE` | Public Trust certificate profile name |
| `WINDOWS_SIGNING_PUBLISHER` | Exact full `SignerCertificate.Subject` from Windows |

The release job signs and timestamps the executable before assembling the
installer. The MSI manufacturer comes from the verified signer. The job signs the
MSI, checks both embedded signatures against the configured subject, installs
the MSI on its Windows runner and compares the installed copy with the portable
EXE. Missing settings or failed checks stop the upload.

To inspect a downloaded package, open PowerShell in its extracted directory.

```powershell
Get-AuthenticodeSignature .\app\asterslice.exe, .\org.ritk.snap.msi |
    Select-Object Path, Status, @{Name='Publisher';Expression={$_.SignerCertificate.Subject}}
```

Both rows should show `Valid` and the expected publisher. A new download can
still trigger a separate SmartScreen reputation warning.
[Microsoft Artifact Signing setup](https://learn.microsoft.com/azure/artifact-signing/quickstart)
and [SmartScreen reputation](https://learn.microsoft.com/windows/apps/package-and-deploy/smartscreen-reputation).
