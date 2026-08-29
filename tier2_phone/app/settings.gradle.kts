// MIRAGE Tier-2 NPU app - Gradle settings.
// SCAFFOLD (not a built/tested APK). Open this folder in Android Studio (Giraffe+/Koala+)
// and let it sync; that generates the Gradle wrapper jar if you don't run `gradle wrapper` first.
pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "MirageNpuTier2"
include(":app")
