use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(super) trait Components {
    fn are_finite(&self) -> bool;
}

impl Components for f32 {
    fn are_finite(&self) -> bool {
        self.is_finite()
    }
}

impl<T: Components> Components for [T] {
    fn are_finite(&self) -> bool {
        self.iter().all(Components::are_finite)
    }
}

impl<T: Components, const N: usize> Components for [T; N] {
    fn are_finite(&self) -> bool {
        self.as_slice().are_finite()
    }
}

impl<T: Components> Components for Vec<T> {
    fn are_finite(&self) -> bool {
        self.as_slice().are_finite()
    }
}

pub(super) fn serialize<T: Serialize + Components, S: Serializer>(
    value: &T,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if !value.are_finite() {
        return Err(serde::ser::Error::custom(
            "annotation numeric components must be finite",
        ));
    }
    value.serialize(serializer)
}

pub(super) fn deserialize<'de, T: Deserialize<'de> + Components, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    let value = T::deserialize(deserializer)?;
    if !value.are_finite() {
        return Err(serde::de::Error::custom(
            "annotation numeric components must be finite",
        ));
    }
    Ok(value)
}
