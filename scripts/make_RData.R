library(readr)

pred_df <- read_csv("python_preds_torch.csv")

print(head(pred_df))
print(str(pred_df))

mclass_pred <- pred_df

save(mclass_pred, file = "mclass_pred.RData")

cat("\nSaved mclass_pred.RData\n")