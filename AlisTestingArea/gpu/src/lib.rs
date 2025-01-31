
use wgpu::{util::DeviceExt, Adapter, Device, Instance, Queue};
use futures_intrusive::channel::shared::oneshot_channel;


pub struct WgpuTask {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue

}


impl WgpuTask {
    pub async fn new() -> Self {

        let instance = wgpu::Instance::default();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();


        WgpuTask{instance, adapter, device, queue}
    }
}