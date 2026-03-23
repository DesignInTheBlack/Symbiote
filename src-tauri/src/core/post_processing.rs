use once_cell::sync::Lazy;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

type Job = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + 'static>;

const POST_PROCESSING_QUEUE_CAPACITY: usize = 64;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum JobPriority {
    Critical,
    BestEffort,
}

impl JobPriority {
    fn weight(&self) -> u8 {
        match self {
            JobPriority::Critical => 2,
            JobPriority::BestEffort => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BackgroundJobHandle {
    cancelled: Arc<AtomicBool>,
}

impl BackgroundJobHandle {
    pub fn cancel(&self) {
        self.cancelled.store(true, AtomicOrdering::SeqCst);
    }
}

struct JobEntry {
    priority: JobPriority,
    seq: u64,
    cancelled: Arc<AtomicBool>,
    job: Job,
}

impl Eq for JobEntry {}

impl PartialEq for JobEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl Ord for JobEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .weight()
            .cmp(&other.priority.weight())
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for JobEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct BackgroundQueue {
    heap: Mutex<BinaryHeap<JobEntry>>,
    notify: Notify,
    len: AtomicUsize,
    seq: AtomicU64,
}

impl BackgroundQueue {
    fn new() -> Arc<Self> {
        let queue = Arc::new(Self {
            heap: Mutex::new(BinaryHeap::new()),
            notify: Notify::new(),
            len: AtomicUsize::new(0),
            seq: AtomicU64::new(0),
        });
        let worker = Arc::clone(&queue);
        tokio::spawn(async move {
            worker.run().await;
        });
        queue
    }

    async fn run(self: Arc<Self>) {
        loop {
            let next = {
                let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
                heap.pop()
            };
            if let Some(entry) = next {
                self.len.fetch_sub(1, AtomicOrdering::SeqCst);
                if !entry.cancelled.load(AtomicOrdering::SeqCst) {
                    (entry.job)().await;
                }
                continue;
            }
            self.notify.notified().await;
        }
    }

    fn enqueue<F, Fut>(&self, priority: JobPriority, job: F) -> Option<BackgroundJobHandle>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let current = self.len.load(AtomicOrdering::SeqCst);
        if current >= POST_PROCESSING_QUEUE_CAPACITY {
            if priority == JobPriority::BestEffort {
                return None;
            }
            // Try to evict one best-effort entry to make room for critical work.
            let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
            let mut entries: Vec<JobEntry> = heap.drain().collect();
            let mut evicted = false;
            if let Some(pos) = entries
                .iter()
                .rposition(|entry| entry.priority == JobPriority::BestEffort)
            {
                entries.remove(pos);
                evicted = true;
                self.len.fetch_sub(1, AtomicOrdering::SeqCst);
            }
            for entry in entries {
                heap.push(entry);
            }
            drop(heap);
            if !evicted {
                return None;
            }
        }
        let cancelled = Arc::new(AtomicBool::new(false));
        let seq = self.seq.fetch_add(1, AtomicOrdering::SeqCst);
        let wrapped: Job = Box::new(move || Box::pin(job()));
        let entry = JobEntry {
            priority,
            seq,
            cancelled: Arc::clone(&cancelled),
            job: wrapped,
        };
        let mut heap = self.heap.lock().unwrap_or_else(|e| e.into_inner());
        heap.push(entry);
        self.len.fetch_add(1, AtomicOrdering::SeqCst);
        drop(heap);
        self.notify.notify_one();
        Some(BackgroundJobHandle { cancelled })
    }
}

static POST_PROCESSING_QUEUE: Lazy<Arc<BackgroundQueue>> = Lazy::new(BackgroundQueue::new);

pub fn enqueue_post_processing_with_priority<F, Fut>(
    priority: JobPriority,
    job: F,
) -> Option<BackgroundJobHandle>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    POST_PROCESSING_QUEUE.enqueue(priority, job)
}

pub fn enqueue_post_processing<F, Fut>(job: F) -> bool
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    enqueue_post_processing_with_priority(JobPriority::BestEffort, job).is_some()
}
