// CI/CD pipeline for the vial impurity classifier.
//
// Lints and tests the code, trains the model, compares it with the Production model,
// validates it, and builds and smoke-tests the serving image. GitHub Actions calls
// these functions; locally they run the same way, e.g. `dagger call lint` or
// `dagger call train --epochs 1 --limit 20 outputs export --path out`.
//
// Registry push and hosting live in hosting.go and are parked until the team decides
// where to host.
//
// The training and serving images are built from Dockerfiles that live in docker/
// (owned by the training and serving parts of the team). See the README's Pipeline
// section for what those Dockerfiles must provide.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"dagger/vial-pipeline/internal/dagger"
)

const (
	pythonImage = "python:3.13.3-slim"
	uvImage     = "ghcr.io/astral-sh/uv:0.9.8"
	curlImage   = "curlimages/curl:8.23.0"

	workdir         = "/app"
	trainDockerfile = "docker/Dockerfile.train"
	serveDockerfile = "docker/Dockerfile.model-serve"
	servingPort     = 5001
	testImagesDir   = "data/raw/images/test"
	// Where the pipeline's own scripts and fixtures are put inside containers
	checksDir = "/opt/checks"
)

type VialPipeline struct {
	// The repository (without local outputs, virtualenvs and downloaded images)
	Source *dagger.Directory
	// Where to publish and deploy (set with with-azure; none means no deployment)
	Azure *AzureTarget
}

func New(
	// Repository root
	// +defaultPath="/"
	// +ignore=[".git", ".venv", "**/__pycache__", "**/.pytest_cache", ".ruff_cache", ".dvc/tmp", "artifacts", "mlflow-state", "models/model", "models/*.json", "validation", "data/raw/images/**/*.jpg", "data/raw/images/**/metadata.csv", "reference", "docs/superpowers", "dagger", "dagger.json"]
	source *dagger.Directory,
) *VialPipeline {
	return &VialPipeline{Source: source}
}

// ---------------------------------------------------------------------------------
// Checks: lint, type check, tests
// ---------------------------------------------------------------------------------

// Python 3.13 + uv with the locked dependencies (dev tools included) and the project installed
func (m *VialPipeline) DevEnv() *dagger.Container {
	return dag.Container().
		From(pythonImage).
		WithFile("/usr/local/bin/uv", dag.Container().From(uvImage).File("/uv")).
		WithMountedCache("/root/.cache/uv", dag.CacheVolume("uv-cache")).
		WithEnvVariable("UV_LINK_MODE", "copy").
		WithEnvVariable("UV_PYTHON_DOWNLOADS", "never").
		WithWorkdir(workdir).
		// Dependencies first, in their own cached layer: only pyproject/uv.lock invalidate it
		WithFile("pyproject.toml", m.Source.File("pyproject.toml")).
		WithFile("uv.lock", m.Source.File("uv.lock")).
		WithExec([]string{"uv", "sync", "--frozen", "--no-install-project"}).
		WithDirectory(".", m.Source).
		WithExec([]string{"uv", "sync", "--frozen"})
}

// Check formatting and lint rules (ruff)
func (m *VialPipeline) Lint(ctx context.Context) (string, error) {
	return m.DevEnv().
		WithExec([]string{"uv", "run", "--frozen", "ruff", "format", "--check"}).
		WithExec([]string{"uv", "run", "--frozen", "ruff", "check"}).
		Stdout(ctx)
}

// Check types (pyright, configured in pyproject.toml)
func (m *VialPipeline) Typecheck(ctx context.Context) (string, error) {
	return m.DevEnv().
		WithExec([]string{"uv", "run", "--frozen", "pyright"}).
		Stdout(ctx)
}

// Run the fast test suite (pytest, no network or data needed)
func (m *VialPipeline) Test(ctx context.Context) (string, error) {
	return m.DevEnv().
		WithExec([]string{"uv", "run", "--frozen", "pytest", "-q", "-m", "not slow"}).
		Stdout(ctx)
}

// Run lint, type check and tests; fail on the first one that fails
func (m *VialPipeline) Check(ctx context.Context) (string, error) {
	steps := []struct {
		name string
		run  func(context.Context) (string, error)
	}{
		{"lint", m.Lint},
		{"typecheck", m.Typecheck},
		{"test", m.Test},
	}
	var report strings.Builder
	for _, step := range steps {
		out, err := step.run(ctx)
		if err != nil {
			return "", fmt.Errorf("%s failed: %w", step.name, err)
		}
		fmt.Fprintf(&report, "== %s\n%s\n", step.name, strings.TrimSpace(out))
	}
	return report.String(), nil
}

// ---------------------------------------------------------------------------------
// Training container
// ---------------------------------------------------------------------------------

// The training image, built from docker/Dockerfile.train
func (m *VialPipeline) TrainImage() *dagger.Container {
	return m.Source.
		DockerBuild(dagger.DirectoryDockerBuildOpts{Dockerfile: trainDockerfile}).
		WithWorkdir(workdir)
}

// How often to run the image download before giving up
const downloadAttempts = 3

// Run the data download (vial_classifier.dataset) until it succeeds. The object store
// occasionally times out on single files; the download skips files it already has, so
// a retry only fetches what is still missing.
func downloadData(ctx context.Context, ctr *dagger.Container) (*dagger.Container, error) {
	download := []string{"python", "-m", "vial_classifier.dataset"}
	for attempt := 1; attempt <= downloadAttempts; attempt++ {
		ctr = ctr.
			WithEnvVariable("DOWNLOAD_ATTEMPT", strconv.Itoa(attempt)).
			WithExec(download, dagger.ContainerWithExecOpts{Expect: dagger.ReturnTypeAny})
		code, err := ctr.ExitCode(ctx)
		if err != nil {
			return nil, err
		}
		if code == 0 {
			return ctr, nil
		}
		fmt.Printf("Image download incomplete (attempt %d/%d)\n", attempt, downloadAttempts)
	}
	return nil, fmt.Errorf("image download failed after %d attempts", downloadAttempts)
}

// Download the data and train the model (notebook 01). Returns artifacts/, mlflow-state/
// and the held-out test images (data/raw/images/test) for the next steps.
func (m *VialPipeline) Train(
	ctx context.Context,
	// MLflow state (the mlflow-state/ directory) of the previous main run.
	// Omit it to start with an empty registry.
	// +optional
	state *dagger.Directory,
	// Train for this many epochs instead of the configured default (quick runs only)
	// +optional
	epochs int,
	// Use only this many training images, at least 14 (quick runs only)
	// +optional
	limit int,
) (*dagger.Directory, error) {
	trainCmd := []string{"python", "-m", "vial_classifier.modeling.train"}
	if epochs > 0 {
		trainCmd = append(trainCmd, "--epochs", strconv.Itoa(epochs))
	}
	if limit > 0 {
		trainCmd = append(trainCmd, "--limit", strconv.Itoa(limit))
	}

	ctr := m.TrainImage()
	if state != nil {
		// Always the same absolute path: MLflow stores absolute artifact locations
		ctr = ctr.WithDirectory(workdir+"/mlflow-state", state)
	}
	ctr, err := downloadData(ctx, ctr)
	if err != nil {
		return nil, err
	}
	ctr = ctr.WithExec(trainCmd)

	return dag.Directory().
		WithDirectory("artifacts", ctr.Directory(workdir+"/artifacts")).
		WithDirectory("mlflow-state", ctr.Directory(workdir+"/mlflow-state")).
		WithDirectory(testImagesDir, ctr.Directory(workdir+"/"+testImagesDir)), nil
}

// ---------------------------------------------------------------------------------
// Deployment container: compare the trained model with the deployed one
// ---------------------------------------------------------------------------------

// The container that runs the deployment logic (notebook 02). There is no separate
// deployment Dockerfile: the training image has every dependency the registry needs.
func (m *VialPipeline) DeployImage() *dagger.Container {
	return m.TrainImage()
}

// Compare the trained model with the Production (deployed) model, promote it if it is
// better, export the Production model and validate it (notebook 02).
//
// Takes the output of Train and returns models/model, artifacts/, mlflow-state/,
// validation/, the test images and decision.json ({"promoted": true|false, ...}).
func (m *VialPipeline) Compare(
	// Output directory of Train
	trained *dagger.Directory,
) *dagger.Directory {
	validate := []string{
		"python", checksDir + "/scripts/validate_model.py",
		"--model-dir", workdir + "/models/model",
		"--images-dir", workdir + "/" + testImagesDir,
		"--expected", checksDir + "/expected_predictions.csv",
		"--output-dir", workdir + "/validation",
	}

	ctr := m.DeployImage().
		WithDirectory(workdir+"/artifacts", trained.Directory("artifacts")).
		WithDirectory(workdir+"/mlflow-state", trained.Directory("mlflow-state")).
		WithDirectory(workdir+"/"+testImagesDir, trained.Directory(testImagesDir)).
		WithDirectory(checksDir+"/scripts", m.Source.Directory("scripts")).
		WithFile(checksDir+"/expected_predictions.csv", m.Source.File("tests/fixtures/expected_predictions.csv")).
		// Registry step of notebook 02, recording whether the model was promoted
		WithExec([]string{"python", checksDir + "/scripts/compare_models.py", "--decision", workdir + "/decision.json"}).
		// Validation never aborts here: its result lands in validation/ and Run decides,
		// so the outputs can still be exported and reported. It runs from a scratch
		// directory because the serving wrapper writes its inference log relative to the
		// working directory, which must not end up in artifacts/.
		WithDirectory(workdir+"/validation", dag.Directory()).
		WithDirectory("/tmp/validate", dag.Directory()).
		WithWorkdir("/tmp/validate").
		WithExec(validate, dagger.ContainerWithExecOpts{
			Expect:         dagger.ReturnTypeAny,
			RedirectStdout: workdir + "/validation/validation.log",
			RedirectStderr: workdir + "/validation/validation.err",
		}).
		WithWorkdir(workdir)

	return dag.Directory().
		WithDirectory("models/model", ctr.Directory(workdir+"/models/model")).
		WithDirectory("artifacts", ctr.Directory(workdir+"/artifacts")).
		WithDirectory("mlflow-state", ctr.Directory(workdir+"/mlflow-state")).
		WithDirectory("validation", ctr.Directory(workdir+"/validation")).
		WithDirectory(testImagesDir, ctr.Directory(workdir+"/"+testImagesDir)).
		WithFile("decision.json", ctr.File(workdir+"/decision.json"))
}

// ---------------------------------------------------------------------------------
// Serving image
// ---------------------------------------------------------------------------------

// The serving image: built from docker/Dockerfile.model-serve, with the exported model
// and its drift baseline added at the paths the serving app reads
func (m *VialPipeline) ServeImage(
	// Exported MLflow pyfunc model (models/model)
	model *dagger.Directory,
	// Drift baseline of that model (drift_baseline_stats.json)
	baseline *dagger.File,
	// Target platform (Azure Container Apps runs linux/amd64)
	// +default="linux/amd64"
	platform dagger.Platform,
) *dagger.Container {
	return m.Source.
		DockerBuild(dagger.DirectoryDockerBuildOpts{
			Dockerfile: serveDockerfile,
			Platform:   platform,
		}).
		WithDirectory(workdir+"/models/model", model).
		WithFile(workdir+"/artifacts/drift_baseline_stats.json", baseline)
}

// Start a serving image and check its endpoints with scripts/smoke_test.py
func (m *VialPipeline) SmokeTest(
	ctx context.Context,
	// Serving image to test
	image *dagger.Container,
	// Test images to send (data/raw/images/test, with its metadata.csv)
	testImages *dagger.Directory,
) (string, error) {
	service := image.
		WithExposedPort(servingPort).
		AsService(dagger.ContainerAsServiceOpts{UseEntrypoint: true})

	// The serving image has requests and the package installed, so it doubles as the client
	return image.
		WithDirectory(checksDir+"/scripts", m.Source.Directory("scripts")).
		WithDirectory("/tmp/smoke-test/images", testImages).
		WithServiceBinding("model", service).
		WithExec([]string{
			"python", checksDir + "/scripts/smoke_test.py",
			"--base-url", fmt.Sprintf("http://model:%d", servingPort),
			"--images-dir", "/tmp/smoke-test/images",
		}).
		Stdout(ctx)
}

// ---------------------------------------------------------------------------------
// The whole pipeline
// ---------------------------------------------------------------------------------

// The decision written by Compare
type decision struct {
	Promoted bool   `json:"promoted"`
	Before   string `json:"production_version_before"`
	After    string `json:"production_version_after"`
}

// Fail unless validation ran and the model reproduced the expected predictions
func checkValidation(ctx context.Context, compared *dagger.Directory) error {
	stdout, _ := compared.File("validation/validation.log").Contents(ctx)
	stderr, _ := compared.File("validation/validation.err").Contents(ctx)
	log := stdout + stderr
	raw, err := compared.File("validation/result.json").Contents(ctx)
	if err != nil {
		return fmt.Errorf("model validation did not run:\n%s", log)
	}
	var result struct {
		Passed bool `json:"passed"`
	}
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		return fmt.Errorf("unreadable validation result: %w", err)
	}
	if !result.Passed {
		return fmt.Errorf("model validation failed:\n%s", log)
	}
	return nil
}

// Check, train, compare with the deployed model, validate, and deploy if the new model
// is better (or if forceDeploy is set) and a target was set with with-azure.
// Returns the outputs of Compare plus deployment.json (export them with
// `export --path out`).
func (m *VialPipeline) Run(
	ctx context.Context,
	// MLflow state (the mlflow-state/ directory) of the previous main run.
	// Omit it to start with an empty registry.
	// +optional
	state *dagger.Directory,
	// Train for this many epochs instead of the configured default (quick runs only)
	// +optional
	epochs int,
	// Use only this many training images, at least 14 (quick runs only)
	// +optional
	limit int,
	// Deploy even if the model is not better (e.g. a hotfix to the serving code)
	// +optional
	forceDeploy bool,
) (*dagger.Directory, error) {
	if _, err := m.Check(ctx); err != nil {
		return nil, err
	}

	trained, err := m.Train(ctx, state, epochs, limit)
	if err != nil {
		return nil, err
	}
	compared := m.Compare(trained)

	if err := checkValidation(ctx, compared); err != nil {
		return nil, err
	}

	raw, err := compared.File("decision.json").Contents(ctx)
	if err != nil {
		return nil, err
	}
	var d decision
	if err := json.Unmarshal([]byte(raw), &d); err != nil {
		return nil, fmt.Errorf("unreadable decision.json: %w", err)
	}

	switch {
	case d.Promoted && d.Before == "":
		fmt.Printf("No model in production yet: version %s becomes Production.\n", d.After)
	case d.Promoted:
		fmt.Printf("New model is better: version %s replaces version %s as Production.\n", d.After, d.Before)
	case forceDeploy:
		fmt.Printf("Production model (version %s) is kept, but a deploy was forced.\n", d.After)
	default:
		fmt.Printf("Production model (version %s) is at least as good: nothing to deploy.\n", d.After)
		return compared, nil
	}
	if m.Azure == nil {
		fmt.Println("No deployment target set (with-azure): skipping the deploy.")
		return compared, nil
	}

	// models/model is always the Production model. Training is deterministic, so this
	// run's drift baseline matches it even when a forced deploy keeps an older version.
	image, err := m.Publish(
		ctx,
		compared.Directory("models/model"),
		compared.File("artifacts/drift_baseline_stats.json"),
		compared.Directory(testImagesDir),
	)
	if err != nil {
		return nil, err
	}
	url, err := m.Deploy(ctx, image)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Deployed %s at %s\n", image, url)

	deployment, err := json.Marshal(map[string]string{"image": image, "url": url})
	if err != nil {
		return nil, err
	}
	return compared.WithNewFile("deployment.json", string(deployment)+"\n"), nil
}
