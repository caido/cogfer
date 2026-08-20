use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use caido_ai::{Error, ErrorKind, Result, TokenStore};

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(crate) struct TokenFile {
    path: PathBuf,
}

impl TokenFile {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) async fn save<T>(&self, tokens: &T) -> Result<()>
    where
        T: serde::Serialize + Sync,
    {
        let data = serde_json::to_vec_pretty(tokens).map_err(|source| {
            Error::new(ErrorKind::Transport, "failed to serialize OAuth tokens").with_source(source)
        })?;
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || save_file(&path, &data))
            .await
            .map_err(|source| {
                Error::new(ErrorKind::Transport, "OAuth token persistence task failed")
                    .with_source(source)
            })?
    }
}

fn save_file(path: &Path, data: &[u8]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(storage_error)?;
    }
    let temporary = temporary_path(path);
    let result = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temporary).map_err(storage_error)?;
        file.write_all(data).map_err(storage_error)?;
        file.sync_all().map_err(storage_error)?;
        drop(file);
        std::fs::rename(&temporary, path).map_err(storage_error)?;
        sync_parent(path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result
}

fn temporary_path(path: &Path) -> PathBuf {
    let id = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("{}.{}.tmp", std::process::id(), id))
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<()> {
    let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    else {
        return Ok(());
    };
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(storage_error)
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<()> {
    Ok(())
}

#[async_trait::async_trait]
impl<T> TokenStore<T> for TokenFile
where
    T: serde::Serialize + Send + Sync,
{
    async fn save(&self, tokens: &T) -> Result<()> {
        TokenFile::save(self, tokens).await
    }
}

fn storage_error(source: std::io::Error) -> Error {
    Error::new(ErrorKind::Transport, "failed to persist OAuth tokens").with_source(source)
}
