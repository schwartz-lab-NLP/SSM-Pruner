import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    """
    A Trainer subclass for knowledge distillation that computes a distillation loss using a soft
    cross-entropy loss (instead of the standard KL-divergence loss).

    The overall loss is computed as:

        loss = (1 - alpha) * student_loss + alpha * soft_ce(student_logits, teacher_logits, T)

    where:
        - student_loss is the original loss returned by the student model (if labels are provided),
        - soft_ce(...) is defined as:

              teacher_probs = softmax(teacher_logits / T, dim=-1)
              log_student_probs = log_softmax(student_logits / T, dim=-1)
              loss = -(teacher_probs * log_student_probs).sum(dim=-1).mean()
              loss = loss * (T ** 2)

        - alpha is the weighting factor for the distillation loss,
        - T is the temperature.

    Args:
        teacher_model (nn.Module): The teacher model for providing soft targets.
        alpha (float, optional): The weight for the distillation loss (default: 0.5).
        temperature (float, optional): The temperature for softening logits (default: 2.0).
        *args, **kwargs: Other arguments passed to the base Trainer.
    """

    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

        if self.teacher_model is not None:
            # Freeze teacher parameters and set teacher to evaluation mode.
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the combined distillation loss.

        The student model is run normally. Then, the teacher model is run (without gradients)
        to obtain teacher logits. A soft cross-entropy loss is computed between the softened logits,
        and then combined with the student’s supervised loss (if available) using the factor alpha.

        Returns:
            Either the loss (if return_outputs is False) or a tuple (loss, outputs).
        """
        labels = inputs.get("labels", None)

        # Forward pass through the student model.
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            student_loss = outputs.get("loss", None)
            student_logits = outputs.get("logits", None)
        else:
            student_loss, student_logits = outputs[0], outputs[1]

        # If no teacher is provided, simply return the student's loss.
        if self.teacher_model is None:
            return (student_loss, outputs) if return_outputs else student_loss

        # Run the teacher model in no-grad mode.
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        if isinstance(teacher_outputs, dict):
            teacher_logits = teacher_outputs.get("logits", None)
        else:
            teacher_logits = teacher_outputs[1]

        T = self.temperature

        # Define the soft cross-entropy loss function.
        def soft_ce(student_logits, teacher_logits, temperature):
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            log_student_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss = -(teacher_probs * log_student_probs).sum(dim=-1).mean()
            return loss * (temperature ** 2)

        kd_loss = soft_ce(student_logits, teacher_logits, T)

        # Combine the student loss and the distillation loss.
        if labels is not None and student_loss is not None:
            loss = (1 - self.alpha) * student_loss + self.alpha * kd_loss
        else:
            loss = kd_loss

        return (loss, outputs) if return_outputs else loss