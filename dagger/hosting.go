// Registry push and deployment: Azure Container Registry (ACR) + Azure Container Apps.
//
// The Azure resources are created once by hand (see infra/azure/README.md). The
// pipeline logs in with a service principal whose credentials live in GitHub secrets.
package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"dagger/vial-pipeline/internal/dagger"
)

const (
	azureCLIImage  = "mcr.microsoft.com/azure-cli:2.93.0"
	serveImageRepo = "model-serve"
)

// Where to publish and deploy the serving image
type AzureTarget struct {
	// ACR login server, e.g. bkkzrt.azurecr.io
	Registry string
	// Resource group that holds the Container App
	ResourceGroup string
	// Name of the Container App
	ContainerApp string
	// Image tag, e.g. sha-<12 chars of the commit>
	Tag string
	// Service principal
	TenantID       string
	SubscriptionID string
	ClientID       string
	ClientSecret   *dagger.Secret
}

// Set the Azure deployment target used by Publish, Deploy and Run
func (m *VialPipeline) WithAzure(
	// ACR login server, e.g. bkkzrt.azurecr.io
	registry string,
	// Resource group that holds the Container App
	resourceGroup string,
	// Name of the Container App
	containerApp string,
	// Service principal: Microsoft Entra tenant ID
	tenantId string,
	// Service principal: Azure subscription ID
	subscriptionId string,
	// Service principal: application (client) ID
	clientId string,
	// Service principal: client secret
	clientSecret *dagger.Secret,
	// Image tag, e.g. sha-<12 chars of the commit>
	// +default="latest"
	tag string,
) *VialPipeline {
	m.Azure = &AzureTarget{
		Registry:       registry,
		ResourceGroup:  resourceGroup,
		ContainerApp:   containerApp,
		Tag:            tag,
		TenantID:       tenantId,
		SubscriptionID: subscriptionId,
		ClientID:       clientId,
		ClientSecret:   clientSecret,
	}
	return m
}

// Build the serving image (linux/amd64), smoke-test it and push it to ACR.
// Returns the pushed image reference including its digest.
func (m *VialPipeline) Publish(
	ctx context.Context,
	// Exported MLflow pyfunc model (models/model)
	model *dagger.Directory,
	// Drift baseline of that model (drift_baseline_stats.json)
	baseline *dagger.File,
	// Test images for the smoke test (data/raw/images/test, with its metadata.csv)
	testImages *dagger.Directory,
) (string, error) {
	if m.Azure == nil {
		return "", fmt.Errorf("no deployment target: call with-azure first")
	}
	image := m.ServeImage(model, baseline, "linux/amd64")
	report, err := m.SmokeTest(ctx, image, testImages)
	if err != nil {
		return "", fmt.Errorf("smoke test failed, image not published: %w", err)
	}
	fmt.Println(report)

	ref := fmt.Sprintf("%s/%s:%s", m.Azure.Registry, serveImageRepo, m.Azure.Tag)
	// ACR accepts the service principal's client ID and secret as registry credentials
	return image.
		WithRegistryAuth(m.Azure.Registry, m.Azure.ClientID, m.Azure.ClientSecret).
		Publish(ctx, ref)
}

// Point the Container App at an image, wait until the new revision is ready and
// return the app's public URL once /health answers
func (m *VialPipeline) Deploy(
	ctx context.Context,
	// Image reference to deploy (as returned by Publish)
	image string,
) (string, error) {
	if m.Azure == nil {
		return "", fmt.Errorf("no deployment target: call with-azure first")
	}
	a := m.Azure
	fqdn, err := dag.Container().
		From(azureCLIImage).
		WithEnvVariable("AZURE_TENANT_ID", a.TenantID).
		WithEnvVariable("AZURE_SUBSCRIPTION_ID", a.SubscriptionID).
		WithEnvVariable("AZURE_CLIENT_ID", a.ClientID).
		WithSecretVariable("AZURE_CLIENT_SECRET", a.ClientSecret).
		WithEnvVariable("RG", a.ResourceGroup).
		WithEnvVariable("APP", a.ContainerApp).
		WithEnvVariable("IMAGE", image).
		WithEnvVariable("PORT", strconv.Itoa(servingPort)).
		// Talks to Azure: never reuse a cached result
		WithEnvVariable("CACHE_BUSTER", strconv.FormatInt(time.Now().UnixNano(), 10)).
		WithNewFile("/tmp/deploy.sh", deployScript).
		WithExec([]string{"bash", "/tmp/deploy.sh"}).
		File("/tmp/fqdn").
		Contents(ctx)
	if err != nil {
		return "", err
	}

	url := "https://" + strings.TrimSpace(fqdn)
	if _, err := checkHealth(ctx, url); err != nil {
		return "", fmt.Errorf("deployed, but %s/health is not ok: %w", url, err)
	}
	return url, nil
}

// Runs in the Azure CLI image. A new image (pushed by digest) always creates a new
// revision; the script waits until that revision is the latest ready one.
const deployScript = `
set -euo pipefail
az config set extension.use_dynamic_install=yes_without_prompt --only-show-errors > /dev/null
az login --service-principal -u "$AZURE_CLIENT_ID" -p "$AZURE_CLIENT_SECRET" \
  --tenant "$AZURE_TENANT_ID" --output none
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# The app is created with a placeholder image; make sure traffic goes to our port
az containerapp ingress update -n "$APP" -g "$RG" --target-port "$PORT" --output none
az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" --output none

latest=""
for i in $(seq 1 60); do
  latest=$(az containerapp show -n "$APP" -g "$RG" --query properties.latestRevisionName -o tsv)
  ready=$(az containerapp show -n "$APP" -g "$RG" --query properties.latestReadyRevisionName -o tsv)
  echo "latest revision: $latest, latest ready revision: $ready"
  if [ "$latest" = "$ready" ]; then break; fi
  if [ "$i" = 60 ]; then echo "revision $latest not ready after 10 minutes"; exit 1; fi
  sleep 10
done

az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv > /tmp/fqdn
echo "Deployed $IMAGE as revision $latest: https://$(cat /tmp/fqdn)"
`

// GET <url>/health of a running service (retries for a minute while the model loads)
func checkHealth(ctx context.Context, url string) (string, error) {
	return dag.Container().
		From(curlImage).
		WithEnvVariable("CACHE_BUSTER", strconv.FormatInt(time.Now().UnixNano(), 10)).
		WithExec([]string{"curl", "-sf", "--retry", "12", "--retry-delay", "5", "--retry-all-errors", url + "/health"}).
		Stdout(ctx)
}
