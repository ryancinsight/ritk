//! VTK scene graph and renderable actor model.
//!
//! A scene S = (A_1, ..., A_n) is an ordered list of actors.
//! Each actor A_i = (data, properties, name, visible).
//! RenderProperties encodes color (RGB in [0,1]), opacity in [0,1],
//! point_size, and line_width.

use crate::domain::vtk_data_object::VtkDataObject;

/// Rendering display properties for a VTK actor.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderProperties {
    pub color: [f32; 3],
    pub opacity: f32,
    pub point_size: f32,
    pub line_width: f32,
}

impl Default for RenderProperties {
    fn default() -> Self {
        Self { color: [1.0, 1.0, 1.0], opacity: 1.0, point_size: 2.0, line_width: 1.0 }
    }
}

/// A renderable actor in a VTK scene.
#[derive(Debug, Clone)]
pub struct VtkActor {
    pub name: String,
    pub data: VtkDataObject,
    pub properties: RenderProperties,
    pub visible: bool,
}

impl VtkActor {
    pub fn new(name: impl Into<String>, data: VtkDataObject) -> Self {
        Self { name: name.into(), data, properties: RenderProperties::default(), visible: true }
    }
    pub fn with_properties(mut self, props: RenderProperties) -> Self {
        self.properties = props; self
    }
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible; self
    }
}

/// VTK scene: an ordered collection of renderable actors.
#[derive(Debug, Default)]
pub struct VtkScene {
    actors: Vec<VtkActor>,
}

impl VtkScene {
    pub fn new() -> Self { Self::default() }

    pub fn add_actor(&mut self, actor: VtkActor) -> &mut Self {
        self.actors.push(actor); self
    }
    pub fn actors(&self) -> &[VtkActor] { &self.actors }

    pub fn actor_by_name(&self, name: &str) -> Option<&VtkActor> {
        self.actors.iter().find(|a| a.name == name)
    }
    pub fn remove_actor(&mut self, name: &str) -> bool {
        if let Some(pos) = self.actors.iter().position(|a| a.name == name) {
            self.actors.remove(pos);
            true
        } else { false }
    }
    pub fn actor_count(&self) -> usize { self.actors.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};

    fn dummy_actor(name: &str) -> VtkActor {
        VtkActor::new(name, VtkDataObject::PolyData(VtkPolyData::default()))
    }

    #[test] fn test_new_scene_empty() {
        let s = VtkScene::new();
        assert_eq!(s.actor_count(), 0);
        assert!(s.actors().is_empty());
    }
    #[test] fn test_add_actor_count() {
        let mut s = VtkScene::new();
        s.add_actor(dummy_actor("actor1"));
        assert_eq!(s.actor_count(), 1);
    }
    #[test] fn test_actor_by_name_found() {
        let mut s = VtkScene::new();
        s.add_actor(dummy_actor("mesh"));
        let a = s.actor_by_name("mesh");
        assert!(a.is_some());
        assert_eq!(a.unwrap().name, "mesh");
    }
    #[test] fn test_actor_by_name_missing() {
        let s = VtkScene::new();
        assert!(s.actor_by_name("missing").is_none());
    }
    #[test] fn test_remove_actor() {
        let mut s = VtkScene::new();
        s.add_actor(dummy_actor("a"));
        assert!(s.remove_actor("a"));
        assert_eq!(s.actor_count(), 0);
        assert!(!s.remove_actor("a"));  // second remove returns false
    }
    #[test] fn test_render_properties_default() {
        let p = RenderProperties::default();
        assert_eq!(p.color, [1.0, 1.0, 1.0]);
        assert!((p.opacity - 1.0).abs() < 1e-6);
        assert!((p.point_size - 2.0).abs() < 1e-6);
        assert!((p.line_width - 1.0).abs() < 1e-6);
    }
    #[test] fn test_multiple_actors_order() {
        let mut s = VtkScene::new();
        s.add_actor(dummy_actor("first"));
        s.add_actor(dummy_actor("second"));
        s.add_actor(dummy_actor("third"));
        let names: Vec<&str> = s.actors().iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
    }
}
