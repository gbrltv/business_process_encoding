library("ECoL")

base_path = "encoding_results"
encodings = list.dirs(path=base_path, full.names=FALSE, recursive=FALSE)
output = paste("dataset_ALL", ".csv", sep="")
written_df = FALSE

for (encoding in encodings) {
  print(encoding)
  path = paste(base_path, "/", encoding, sep="") 
  files = list.files(path, pattern="\\.csv$")  

  aux = 1
  for (file in files){
    print(paste(aux, length(files), sep=" out of "))
    aux = aux + 1
    x = unlist(strsplit(file, "_"))
    
    scenario = x[1]
    size = x[2]
    anomaly = x[3]
    impact = strsplit(x[4], ".csv")[1]
    
    if (anomaly == "normal") {
      next
    }
    
    dataset_raw = read.csv(paste(path, file, sep="/"))
    
    labels_extra = c("case", "time")
    dataset_extra = dataset_raw[,labels_extra]
    
    dataset_features = dataset_raw[,-which(names(dataset_raw) %in% labels_extra)]
    if (encoding == "tokenreplay") {
      dataset_features$trace_is_fit = as.integer(as.logical(dataset_features$trace_is_fit))  
    }
    
    metrics = complexity(label ~ ., dataset_features)
    
    results = data.frame(c(encoding=encoding, file=as.character(file), 
                           scenario=scenario, size=size, anomaly=anomaly, 
                           impact=impact, time=dataset_extra$time[0], 
                           memory=dataset_extra$memory[0], 
                           metrics), stringsAsFactors=FALSE)
    
    if (!written_df) {
      write.table(results, output, sep=",", col.names=colnames(results), row.names=FALSE)
      written_df = TRUE
    } else {
      write.table(results, output, sep=",", append=T, col.names=FALSE, row.names=FALSE)  
    }
  }
}
