from data import get_client_a_dataset, get_client_b_dataset

print("Verifying zero-overlap splits...")
print()

a_train = get_client_a_dataset(train=True)
b_train = get_client_b_dataset(train=True)

a_classes = set(y for _, y in a_train)
b_classes = set(y for _, y in b_train)

print(f"Client A classes: {sorted(a_classes)}")
print(f"Client A size: {len(a_train)}")
print()

print(f"Client B classes: {sorted(b_classes)}")
print(f"Client B size: {len(b_train)}")
print()

overlap = a_classes & b_classes
print(f"Overlap: {sorted(overlap)} {'✓ ZERO OVERLAP!' if len(overlap) == 0 else '✗ STILL HAS OVERLAP'}")
print(f"Total coverage: {sorted(a_classes | b_classes)}")
print()

print("CIFAR-10 class mapping:")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Client A (vehicles): {', '.join(classes[i] for i in sorted(a_classes))}")
print(f"Client B (animals): {', '.join(classes[i] for i in sorted(b_classes))}")
